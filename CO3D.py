import torch.utils.data as data
import numpy as np
from PIL import Image
import torch as th
from co3d.dataset.data_types import load_dataclass_jgzip, FrameAnnotation
from pathlib import Path
from typing import List
from posewarp import Warper
import os
from torchvision import transforms
from torchvision.transforms import functional as TF
from warpback import (
    RGBDRenderer,
    transformation_from_parameters
)
from diffusers import StableDiffusionInpaintPipeline
from random import choice
import math

class WarpCO3DDataset(data.Dataset):
    def __init__(self,
                 root="./data",
                 prompt="a photo of {}",
                 rare_token="xxxsy5zt",
                 offset=1, # code needs to be updated to work with offset > 1
                 use_depth_estimation=False, # Whether to use DepthAnything for when no GT/fitted depth is available
                 use_random_warp=False, # Perform random forward-backward warp, only when use_depth_estimation=True
                 trans_range={"x":0.2, "y":-1, "z":-1, "a":-1, "b":-1, "c":-1},
                 device='cuda:0'
    ):
        if not isinstance(root, Path):
            root = Path(root)

        # potentially pass a prompt list like ["a photo of [V]", "a rendering of [V]"] and choose one at random
        if type(prompt) == str:
            self.prompt = [prompt]
        else:
            self.prompt = prompt

        self.transform = transforms.Compose(
            [
                transforms.ToTensor()
            ]
        )
        self.rare_token = rare_token
        self.root = root
        self.offset = offset
        self.use_depth_estimation = use_depth_estimation
        self.use_random_warp = use_random_warp
        self.trans_range = trans_range
        self.device = device

        # For now, just use off-the-shelf SD1.5 inpaint
        # A fine-tuned model should probably be used in the future
        self.inpaint_pipeline = StableDiffusionInpaintPipeline.from_pretrained(
            'runwayml/stable-diffusion-inpainting',
            torch_dtype=th.float16,
            safety_checker=None
        ).to(device)

        self.inpaint_pipeline.set_progress_bar_config(leave=False)
        self.inpaint_pipeline.set_progress_bar_config(disable=True)

        if use_depth_estimation:
            from transformers import pipeline
            self.depth_estimator = pipeline(task="depth-estimation", model="LiheYoung/depth-anything-small-hf", device=self.device)
            self.renderer = RGBDRenderer(device)
        else:
            self.warper = Warper()

        num_total_images = 0
        sequences = []
        for item in root.iterdir():
            if not item.is_dir():
                continue

            annotations_path = item.joinpath('frame_annotations.jgz')

            if not annotations_path.exists():
                continue

            frame_annotations = load_dataclass_jgzip(annotations_path, List[FrameAnnotation])

            for subdir in item.iterdir():
                if not subdir.is_dir():
                    continue

                if 'pointcloud' in map(lambda x: x.stem, subdir.iterdir()):
                    annotations = list(filter(lambda frame: frame.sequence_name == subdir.stem, frame_annotations))
                    annotations.sort(key=lambda frame: frame.frame_number)

                    sequences.append(annotations)

                    num_total_images += len(annotations) - 1

        self.sequences = sequences
        self.size = num_total_images

    def co3d_annotation_to_opencv_pose(self, entry: FrameAnnotation):
        p = entry.viewpoint.principal_point
        f = entry.viewpoint.focal_length
        h, w = entry.image.size
        K = np.eye(3)
        s = min(w,h)
        K[0, 0] = f[0] * s / 2
        K[1, 1] = f[1] * s / 2
        K[0, 2] = w / 2 - p[0] * s / 2
        K[1, 2] = h / 2 - p[1] * s / 2

        R = np.asarray(entry.viewpoint.R).T
        T = np.asarray(entry.viewpoint.T)
        pose = np.concatenate([R, T[:,None]], 1)
        pose = np.diag([-1,-1,1]).astype(np.float32) @ pose

        return K, pose

    def _load_16big_png_image(self, depth_png_path: str | Path, crop=None):
        # CO3D stores depth as float16 binary representation as uint16 and saves as PNG16

        with Image.open(depth_png_path) as depth_pil:
            if crop is not None:
                depth_pil = TF.crop(depth_pil, *crop)
            depth = np.frombuffer(np.array(depth_pil, dtype=np.uint16), dtype=np.float16)
            depth = depth.astype(np.float32)
            depth = depth.reshape(*depth_pil.size[::-1])

        return depth

    def _depth_to_colormap(self, depth_arr: np.ndarray, cmap='inferno'):
        import matplotlib

        heatmap = depth_arr.astype(np.float32)
        heatmap /= heatmap.max()
        heatmap = matplotlib.colormaps[cmap](heatmap)
        heatmap = (heatmap * 255).astype(np.uint8)

        return Image.fromarray(heatmap)

    def rand_tensor(self, r, l):
        # From AdaMPI warpback

        if r < 0:  # we can set a negtive value in self.trans_range to avoid random transformation
            return th.zeros((l, 1, 1))
        rand = th.rand((l, 1, 1))        
        sign = 2 * (th.randn_like(rand) > 0).float() - 1
        return sign * (r / 2 + r / 2 * rand)

    def get_rand_ext(self, bs):
        # From AdaMPI warpback

        x, y, z = self.trans_range['x'], self.trans_range['y'], self.trans_range['z']
        a, b, c = self.trans_range['a'], self.trans_range['b'], self.trans_range['c']
        cix = self.rand_tensor(x, bs)
        ciy = self.rand_tensor(y, bs)
        ciz = self.rand_tensor(z, bs)
        aix = self.rand_tensor(math.pi / a, bs)
        aiy = self.rand_tensor(math.pi / b, bs)
        aiz = self.rand_tensor(math.pi / c, bs)
        
        axisangle = th.cat([aix, aiy, aiz], dim=-1)  # [b,1,3]
        translation = th.cat([cix, ciy, ciz], dim=-1)
        
        cam_ext = transformation_from_parameters(axisangle, translation)  # [b,4,4]
        cam_ext_inv = th.inverse(cam_ext)  # [b,4,4]
        return cam_ext[:, :-1], cam_ext_inv[:, :-1]

    def adawarp(self, rgb: Image, disp, K: np.ndarray | th.Tensor, Rt: np.ndarray | th.Tensor=None, Rt_tar: np.ndarray | th.Tensor=None):
        # Use AdaMPI mesh-based warping method
        assert self.use_depth_estimation

        if type(disp) != th.Tensor:
            disp = transforms.ToTensor()(disp)[None]

        if type(rgb) != th.Tensor:
            rgb = transforms.ToTensor()(rgb)[None]

        if type(Rt) == np.ndarray:
            Rt = th.from_numpy(Rt)[None]
            Rt_tar = th.from_numpy(Rt_tar)[None]

        disp = disp.to(device=self.device, dtype=th.float32)
        rgb = rgb.to(device=self.device, dtype=th.float32)
        Rt = Rt.to(device=self.device, dtype=th.float32)
        Rt_tar = Rt_tar.to(device=self.device, dtype=th.float32)

        rgbd = th.cat([rgb, disp], dim=1)

        mesh = self.renderer.construct_mesh(rgbd, K)

        if self.use_random_warp:
            Rt_render = Rt_tar
        else:
            Rt_render = (Rt_tar @ th.linalg.inv(Rt))[:, :3]
        rgb_warped, *rest = self.renderer.render_mesh(mesh, K, Rt_render)
        rgb_warped = th.clamp(rgb_warped, 0., 1.)

        return rgb_warped, *rest

    def __getitem__(self, index):
        if index >= self.size:
            raise ValueError("out of bounds")

        seen = 0
        for annotations in self.sequences:
            if seen + len(annotations) - 2 >= index:
                break

            seen += len(annotations) - 1

        frame = annotations[index - seen]
        next_frame = annotations[index - seen + self.offset]

        rgb = Image.open(self.root.joinpath(frame.image.path))
        rgb_copy = rgb.copy()

        maybe_resize = transforms.Resize(size=512) if rgb.size[0] < 512 or rgb.size[1] < 512 else lambda x: x
        crop_params = transforms.RandomCrop.get_params(maybe_resize(rgb), (512, 512))

        target = Image.open(self.root.joinpath(next_frame.image.path))
        target = TF.crop(target, *crop_params)

        rgb = TF.crop(rgb, *crop_params)

        K, Rt = self.co3d_annotation_to_opencv_pose(frame)
        K_tar, Rt_tar = self.co3d_annotation_to_opencv_pose(next_frame)

        # Pad 3x4 [R|t] to 4x4 matrix
        Rt = np.vstack((Rt, [0, 0, 0, 1]))
        Rt_tar = np.vstack((Rt_tar, [0, 0, 0, 1]))

        if 'DEBUG_CO3D' in os.environ:
            rgb.save("_debug.png")
            target.save("_debug.png")

        if self.use_depth_estimation:
            # This just uses AdaMPI's warpback module to warp from predicted disparity
            # There is no fitting between GT (if it exists) and recovered depth

            disp = self.depth_estimator(rgb)['depth']

            if self.use_random_warp:
                Rt, Rt_tar = self.get_rand_ext(1)

            # We could pass estimated camera parameters
            K = th.tensor([
                [0.58, 0, 0.5],
                [0, 0.58, 0.5],
                [0, 0, 1]
            ])[None].to(self.device)

            rgb_warped, disp_warped, mask = self.adawarp(rgb, disp, K, Rt, Rt_tar)

            if self.use_random_warp:
                rgb_warped, _, mask = self.adawarp(rgb_warped, disp_warped, K, Rt_tar, Rt)

            rgb_warped = rgb_warped.cpu().permute(0, 2, 3, 1).squeeze(0).numpy().astype(np.float32)
            inpaint_mask = (1 - mask.cpu().repeat(1, 3, 1, 1).permute(0, 2, 3, 1)).squeeze(0).numpy().astype(np.uint8)

            if 'DEBUG_CO3D' in os.environ:
                Image.fromarray((rgb_warped * 255).astype(np.uint8)).save('_debug_warp.png')
                Image.fromarray(inpaint_mask * 255).save('_debug_mask.png')
        else:
            from PIL import ImageFilter
            # Otherwise, use preprocessed predicted depth (adjusted to fit against GT depth)
            rgb = np.array(rgb)

            depth_path = self.root.joinpath(frame.depth.path)
            depth_path = depth_path.parents[1].joinpath('processed', Path(frame.depth.path).name)
            depth = self._load_16big_png_image(depth_path, crop_params)

            rgb_warped, mask, *_ = self.warper.forward_warp(rgb, None, depth, Rt, Rt_tar, K, K_tar)

            # Hacky way to dilate warped edges
            inpaint_mask = (1 - mask[..., np.newaxis].repeat(3, axis=-1).astype(np.uint8))
            inpaint_mask = inpaint_mask.clip(0, 1)
            inpaint_mask = Image.fromarray(255 * inpaint_mask).filter(ImageFilter.MaxFilter(3))
            inpaint_mask = (np.array(inpaint_mask) / 255).astype(np.uint8)

            # Apply dilated mask
            rgb_warped[inpaint_mask > 0.5] = 0

            if 'DEBUG_CO3D' in os.environ:
                self._depth_to_colormap(depth).save("_debug_depth.png")
                Image.fromarray(inpaint_mask * 255).save('_debug_mask.png')
                Image.fromarray(rgb_warped).save('_debug_warp.png')

            rgb_warped = rgb_warped.astype(np.float32) / 255.

        inpaint_mask = Image.fromarray(inpaint_mask * 255)
        rgb_warped = Image.fromarray((rgb_warped * 255).astype(np.uint8))

        inpainted_image = self.inpaint_pipeline(
            prompt="photograph, high resolution, high quality",
            negative_prompt="cartoon",
            image=rgb_warped,
            mask_image=inpaint_mask,
            strength=1.0,
            num_inference_steps=25,
            guidance_scale=1.5
        ).images[0]

        if 'DEBUG_CO3D' in os.environ:
            inpainted_image.save('_debug_inpaint.png')

        prompt = choice(self.prompt).format(self.rare_token)

        # cldm expects images in WxHxC not CxWxH
        return {
            'jpg': (self.transform(target).permute(1, 2, 0) * 2) - 1,
            'hint': (self.transform(inpainted_image).permute(1, 2, 0) * 2) - 1,
            'txt': prompt,
            # 'pil_original': rgb_copy,
            'pil_warp': rgb_warped,
            # 'pil_mask': inpaint_mask,
            # 'pil_target': target,
            # 'pil_condition': inpainted_image
        }
    
    def __len__(self):
        return self.size

if __name__ == "__main__":
    dataset = WarpCO3DDataset()
    data = dataset[0]
    hint = (data['hint'].numpy() + 1) * 127.5
    hint = hint.astype(np.uint8)
    Image.fromarray(hint).save('_debug_fit_sequential_condition.png')
    data['pil_warp'].save('_debug_fit_sequential_warp.png')

    dataset = WarpCO3DDataset(use_depth_estimation=True, use_random_warp=True)
    data = dataset[0]
    hint = (data['hint'].numpy() + 1) * 127.5
    hint = hint.astype(np.uint8)
    Image.fromarray(hint).save('_debug_random_condition.png')
    data['pil_warp'].save('_debug_random_warp.png')
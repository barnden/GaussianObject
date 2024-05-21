from cldm.model import create_model, load_state_dict
from minlora import add_lora, LoRAParametrization
from torch import nn
from functools import partial
from cldm.ddim_hacked import DDIMSampler
import numpy as np
from threestudio.systems.gaussian_object_system import process

if __name__ == '__main__':
    condition = Image.load('./condition.png')
    ckpt_step = 5399
    rare_token = "xxy5syt00"

    controlnet = create_model('models/control_v11f1e_sd15_tile.yaml').cpu()
    controlnet.load_state_dict(load_state_dict('models/v1-5-pruned.ckpt', location='cuda'), strict=False)
    controlnet.load_state_dict(load_state_dict('models/control_v11f1e_sd15_tile.pth', location='cuda'), strict=False)

    lora_config = {
        nn.Embedding: {
            "weight": partial(LoRAParametrization.from_embedding, rank=64)
        },
        nn.Linear: {
            "weight": partial(LoRAParametrization.from_linear, rank=64)
        },
        nn.Conv2d: {
            "weight": partial(LoRAParametrization.from_conv2d, rank=64)
        }
    }

    # Assumption is that we added LoRA to diffusion, ControlNet and CLIP
    for name, module in controlnet.model.diffusion_model.named_modules():
        if name.endswith('transformer_blocks'):
            add_lora(module, lora_config=lora_config)

    for name, module in controlnet.control_model.named_modules():
        if name.endswith('transformer_blocks'):
            add_lora(module, lora_config=lora_config)

    add_lora(controlnet.cond_stage_model, lora_config=lora_config)

    controlnet.load_state_dict(load_state_dict(f'output/controlnet_finetune/co3d/ckpts-lora/lora-step={ckpt_step}.ckpt', location='cuda'), strict=False)
    controlnet = controlnet.cuda()

    ddim_sampler = DDIMSampler(controlnet)

    input_image = np.array(condition)

    result, sds = process(
        controlnet,
        ddim_sampler,
        input_image,
        f"a photo of {rare_token}",
    )

    from PIL import Image

    Image.fromarray(result[0]).save('_result.png')
    condition.save('_input.png')
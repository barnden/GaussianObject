python train_lora.py --exp_name controlnet_finetune/co3d \
    --prompt xxy5syt00 --sh_degree 2 --resolution 4 --sparse_num 4 \
    --sd_locked --train_lora \
    --add_diffusion_lora --add_control_lora --add_clip_lora
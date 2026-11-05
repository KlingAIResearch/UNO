llm_path="Qwen/Qwen2.5-7B-Instruct"
vae_path="black-forest-labs/FLUX.1-dev/ae.safetensors"
vit_path="HuggingFaceM4/siglip-so400m-14-980-flash-attn2-navit"
resume_from="ByteDance-Seed/BAGEL-7B-MoT/"

job_name="job_name"
output_path=/path_to_logging/$job_name
ckpt_path=$output_path/checkpoints

torchrun \
    --nproc_per_node=8 \
    train/pretrain_uno_edit_navit.py \
    --dataset_config_file ./data/configs/uno_example_edit.yaml \
    --layer_module Qwen2MoTDecoderLayer \
    --vae_path $vae_path \
    --vit_path $vit_path \
    --llm_path $llm_path \
    --use_flex True \
    --num_metaqueries 256 \
    --lr 1e-4 \
    --lr_scheduler constant \
    --warmup_steps 500 \
    --total_steps 5000 \
    --ema 0.9999 \
    --ce_weight 0.1 \
    --metaquery_weight 0.2 \
    --mse_weight 1.0 \
    --timestep_shift 4.0 \
    --freeze_vae \
    --freeze_vit \
    --llm_qk_norm True \
    --visual_und True \
    --visual_gen True \
    --wandb_name $job_name \
    --save_every 1000 \
    --results_dir $output_path \
    --checkpoint_dir $ckpt_path \
    --max_latent_size 64  \
    --resume_from $resume_from \
    --finetune_from_ema \
    --resume_model_only True \
    --num_workers 1 # use small num_workers since the num_used_data (10) are not enough to split

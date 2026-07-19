python evaluate_dreamer.py \
    --envs VizdoomBasic-v1 \
    --dreamer_ckpt path/to/world_model.ckpt \
    --policy_ckpt path/to/policy_agent.ckpt \
    --out_dir ./logs/videos \
    --fps 15

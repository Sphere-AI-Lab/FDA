export MKL_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2
export OMP_NUM_THREADS=2

export PYTHONPATH="$PYTHONPATH:$PWD"

CUDA_VISIBLE_DEVICES=3 python construct_fda.py \
      --model roberta-base \
      --name init_by_gauss_random_pos \
      --scale 0.01 \
      --anchor_num 64 \
      --token_num 5 \
      --opt_steps 1200 \
      --opt_lr 1e-2 \
      --model_location /home/shikexuan/MergeLM/save_models_2 \
      --anchor_loss cos \
      --save_dual_anchors

CUDA_VISIBLE_DEVICES=4 python construct_fda.py \
      --model roberta-base \
      --name init_by_weights_pos \
      --scale 0.01 \
      --anchor_num 64 \
      --token_num 5 \
      --opt_steps 1200 \
      --opt_lr 1e-2 \
      --model_location /home/shikexuan/MergeLM/save_models_2 \
      --anchor_loss cos \
      --save_dual_anchors \

CUDA_VISIBLE_DEVICES=5 python construct_fda.py \
      --model roberta-large \
      --name init_by_gauss_random_pos \
      --scale 0.01 \
      --anchor_num 64 \
      --token_num 5 \
      --opt_steps 1200 \
      --opt_lr 1e-2 \
      --anchor_loss cos \
      --save_dual_anchors \

CUDA_VISIBLE_DEVICES=6 python construct_fda.py \
      --model roberta-large \
      --name init_by_weights_pos \
      --scale 0.01 \
      --anchor_num 64 \
      --token_num 5 \
      --opt_steps 1200 \
      --opt_lr 1e-2 \
      --anchor_loss cos \
      --save_dual_anchors
export MKL_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2
export OMP_NUM_THREADS=2

export PYTHONPATH="$PYTHONPATH:$PWD"


CUDA_VISIBLE_DEVICES=0 python construct_fda.py \
      --model ViT-B-32 \
      --name init_by_gauss_random \
      --scale 0.01 \
      --anchor_num 64 \
      --token_num 50 \
      --opt_steps 1200 \
      --opt_lr 1e-2 \
      --anchor_loss cos \
      --save_dual_anchors

CUDA_VISIBLE_DEVICES=1 python construct_fda.py \
      --model ViT-B-32 \
      --name init_by_weights \
      --scale 0.01 \
      --anchor_num 64 \
      --token_num 50 \
      --opt_steps 1200 \
      --opt_lr 1e-2 \
      --anchor_loss cos \
      --save_dual_anchors
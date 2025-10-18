export MKL_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2
export OMP_NUM_THREADS=2

export PYTHONPATH="$PYTHONPATH:$PWD"

CUDA_VISIBLE_DEVICES=5 python adapt_fda.py \
      --model ViT-B-32 \
      --name init_by_gauss_random \
      --init_params pretrained \
      --scale 0.01 \
      --anchor_num 64 \
      --token_num 50 \
      --opt_steps 1200 \
      --opt_lr 1e-2 \
      --anchor_loss cos \
      --adapt_loss cos \
      --read_anchors_path /data/shikexuan/anchors_for_vision \
      --adapt_batch_size 128 \
      --adapt_epochs 100 \
      --adapt_lr 1e-5

# CUDA_VISIBLE_DEVICES=1 python adapt_fda.py \
#       --model ViT-B-32 \
#       --name init_by_gauss_random \
#       --init_params ta \
#       --scale 0.01 \
#       --anchor_num 64 \
#       --token_num 50 \
#       --opt_steps 1200 \
#       --opt_lr 1e-2 \
#       --anchor_loss cos \
#       --adapt_loss cos \
#       --adapt_batch_size 128 \
#       --adapt_epochs 100 \
#       --adapt_lr 1e-2
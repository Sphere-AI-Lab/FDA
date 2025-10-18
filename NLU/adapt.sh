export MKL_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2
export OMP_NUM_THREADS=2

export PYTHONPATH="$PYTHONPATH:$PWD"

# for base model

# init_by_gauss_random_pos
# init_by_params_pos
CUDA_VISIBLE_DEVICES=0 python adapt.py \
      --model roberta-base \
      --name init_by_gauss_random_pos \
      --init_params pretrained \
      --scale 0.01 \
      --anchor_num 64 \
      --token_num 5 \
      --opt_steps 1200 \
      --opt_lr 1e-2 \
      --anchor_loss cos \
      --adapt_lr 5e-5 \
      --adapt_loss mse \
      --save_dual_anchors \
      --adapt_batch_size 128 \
      --adapt_epochs 125


# init_by_gauss_random_pos
# init_by_params_pos
# tsv, wudi,ta
# CUDA_VISIBLE_DEVICES=4 python adapt.py \
#       --model roberta-base \
#       --name init_by_weights_pos \
#       --init_params tsv \
#       --scale 0.01 \
#       --anchor_num 64 \
#       --token_num 5 \
#       --opt_steps 1200 \
#       --opt_lr 1e-2 \
#       --anchor_loss cos \
#       --save_dual_anchors \
#       --adapt_lr 1e-2 \
#       --adapt_loss mse \
#       --adapt_epochs 100



# for roberta-large

# init_by_gauss_random_pos
# init_by_params_pos
# CUDA_VISIBLE_DEVICES=1 python adapt.py \
#       --model roberta-large \
#       --name init_by_gauss_random_pos \
#       --init_params pretrained \
#       --scale 0.01 \
#       --anchor_num 64 \
#       --token_num 5 \
#       --opt_steps 1200 \
#       --opt_lr 1e-2 \
#       --anchor_loss cos \
#       --adapt_lr 5e-5 \
#       --adapt_loss mse \
#       --save_dual_anchors \
#       --adapt_batch_size 128 \
#       --adapt_epochs 25


# init_by_gauss_random_pos
# init_by_params_pos
# tsv, wudi,ta
# CUDA_VISIBLE_DEVICES=0 python adapt.py \
#       --model roberta-large \
#       --name init_by_gauss_random_pos \
#       --init_params ta \
#       --scale 0.01 \
#       --anchor_num 64 \
#       --token_num 5 \
#       --opt_steps 1200 \
#       --opt_lr 1e-2 \
#       --anchor_loss cos \
#       --save_dual_anchors \
#       --adapt_lr 1e-2 \
#       --adapt_loss mse \
#       --adapt_epochs 15

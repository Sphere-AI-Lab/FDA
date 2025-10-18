export MKL_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2
export OMP_NUM_THREADS=2

CUDA_VISIBLE_DEVICES=4 python main_fda_construct.py \
      --name init_by_gauss_random \
      --scale 0.01 \
      --anchor_num 8192 \
      --gen_steps 200 \
      --random \
      --gen_lr 1e-2 \
      --save_dual_anchors \
      --do_math \
      --do_code \

CUDA_VISIBLE_DEVICES=4 python main_fda_construct.py \
      --name init_by_weights \
      --scale 0.01 \
      --anchor_num 8192 \
      --gen_steps 200 \
      --random \
      --gen_lr 1e-2 \
      --save_dual_anchors \
      --do_math \
      --do_code \
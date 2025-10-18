export MKL_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2
export OMP_NUM_THREADS=2

export MKL_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2
export OMP_NUM_THREADS=2

CUDA_VISIBLE_DEVICES=4 python merged_by_FDAs.py \
      --name init_by_gauss_random \
      --scale 0.01 \
      --anchor_num 8192 \
      --gen_steps 200 \
      --gen_lr 1e-2 \
      --do_math \
      --do_code \
      --adapt_lr 1e-2 \
      --merged_model_path /data/shikexuan/save_merged_models_dare/math_code/math/task_arithmetic_scaling_coefficient_0.5 \
      --adapt_batch_size 8192 \
      --adapt_epochs 50 \

CUDA_VISIBLE_DEVICES=4 python merged_by_FDAs.py \
      --name init_by_weights \
      --scale 0.01 \
      --anchor_num 8192 \
      --gen_steps 200 \
      --gen_lr 1e-2 \
      --do_math \
      --do_code \
      --adapt_lr 1e-2 \
      --merged_model_path /data/shikexuan/save_merged_models_dare/math_code/math/task_arithmetic_scaling_coefficient_0.5 \
      --adapt_batch_size 8192 \
      --adapt_epochs 50 \
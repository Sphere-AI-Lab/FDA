export MKL_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2
export OMP_NUM_THREADS=2

CUDA_VISIBLE_DEVICES=4 python eval.py \
      --do_code \
      --gpu 1 \
      --model_path /data/shikexuan/save_merge_models/math_code/init_by_gauss_random-scale-0.01/Llama-2-13b-hf-gen-200-gen-lr-0.01-anchor-8192-align-50-lr-0.01-bs-8192/code \

CUDA_VISIBLE_DEVICES=4 python eval.py \
      --do_math \
      --gpu 1 \
      --model_path /data/shikexuan/save_merge_models/math_code/init_by_gauss_random-scale-0.01/Llama-2-13b-hf-gen-200-gen-lr-0.01-anchor-8192-align-50-lr-0.01-bs-8192/math
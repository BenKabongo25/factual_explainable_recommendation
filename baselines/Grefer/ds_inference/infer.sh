export MODEL_PATH='../ckpts/amazon_grefer'
# mkdir -p $MODEL_PATH/$1
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3  infer.py --model_path $MODEL_PATH \
    --streategy Parallel \
    --batch_size 16 \
    --save_dir ../convert_files/amazon \
# &> $MODEL_PATH/math_graph.log 
#!/bin/bash

# The port for communication. Note that if you want to run multiple tasks on the same machine,
# you need to specify different port numbers.
export MASTER_PORT=6092
export CUDA_VISIBLE_DEVICES=0
export GPUS_PER_NODE=1


########################## Evaluate Refcoco+ ##########################
bpe_dir=/home/shreya/scratch/Regional/polygon-transformer/utils/BPE
user_dir=/home/shreya/scratch/Regional/polygon-transformer/polyformer_module
selected_cols=0,5,6,2,4,3

model='polyformer_b'
num_bins=64
batch_size=16
epoch=100


dataset='regional'
split='test'
ckpt_path=/home/shreya/scratch/Regional/polygon-transformer/polyformer_b_checkpoints/100_5e-5_512/checkpoint_best.pt
data=/home/shreya/scratch/Regional/polygon-transformer/datasets/finetune/Regional_test.tsv
result_path=/home/shreya/scratch/Regional/polygon-transformer/results_${model}/${dataset}/epoch_${epoch}
vis_dir=${result_path}/vis/${split}
result_dir=${result_path}/result/${split}

python3 -m torch.distributed.launch --nproc_per_node=${GPUS_PER_NODE} --master_port=${MASTER_PORT} /home/shreya/scratch/Regional/polygon-transformer/evaluate.py \
    ${data} \
    --path=${ckpt_path} \
    --user-dir=${user_dir} \
    --task=refcoco \
    --batch-size=${batch_size} \
    --log-format=simple --log-interval=10 \
    --seed=7 \
    --gen-subset=${split} \
    --results-path=${result_path} \
    --no-repeat-ngram-size=3 \
    --fp16 \
    --num-workers=0 \
    --num-bins=${num_bins} \
    --vis_dir=${vis_dir} \
    --result_dir=${result_dir} \
    --model-overrides="{\"data\":\"${data}\",\"bpe_dir\":\"${bpe_dir}\",\"selected_cols\":\"${selected_cols}\"}"
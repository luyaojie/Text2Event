#!/usr/bin/env bash
# -*- coding:utf-8 -*-

EXP_ID=$(date +%F-%H-%M-$RANDOM)
export CUDA_VISIBLE_DEVICES="0"
export batch_size="16"
export model_name=t5-base
export data_name=one_ie_ace2005_subtype
export lr=5e-5
export task_name="event"
export seed="421"
export lr_scheduler=constant_with_warmup
export label_smoothing="0"
export epoch=30
export pretrain=5
export decoding_format='tree'
export eval_steps=500
export warmup_steps=0
export constraint_decoding='--constraint_decoding'

OPTS=$(getopt -o b:d:m:i:t:s:l:f: --long batch:,device:,model:,data:,task:,seed:,lr:,lr_scheduler:,label_smoothing:,epoch:,format:,eval_steps:,warmup_steps:,pretrain:,wo_constraint_decoding -n 'parse-options' -- "$@")

if [ $? != 0 ]; then
  echo "Failed parsing options." >&2
  exit 1
fi

eval set -- "$OPTS"

while true; do
  case "$1" in
  -b | --batch)
    batch_size="$2"
    shift
    shift
    ;;
  -d | --device)
    CUDA_VISIBLE_DEVICES="$2"
    shift
    shift
    ;;
  -m | --model)
    model_name="$2"
    shift
    shift
    ;;
  -i | --data)
    data_name="$2"
    shift
    shift
    ;;
  -t | --task)
    task_name="$2"
    shift
    shift
    ;;
  -s | --seed)
    seed="$2"
    shift
    shift
    ;;
  -l | --lr)
    lr="$2"
    shift
    shift
    ;;
  -f | --format)
    decoding_format="$2"
    shift
    shift
    ;;
  --lr_scheduler)
    lr_scheduler="$2"
    shift
    shift
    ;;
  --label_smoothing)
    label_smoothing="$2"
    shift
    shift
    ;;
  --epoch)
    epoch="$2"
    shift
    shift
    ;;
  --eval_steps)
    eval_steps="$2"
    shift
    shift
    ;;
  --warmup_steps)
    warmup_steps="$2"
    shift
    shift
    ;;
  --pretrain)
    pretrain="$2"
    shift
    shift
    ;;
  --wo_constraint_decoding)
    constraint_decoding=""
    shift
    ;;
  --)
    shift
    break
    ;;
  *)
    echo "$1" not recognize.
    exit
    ;;
  esac
done

# google/mt5-base -> google_mt5-base
model_name_log=$(echo ${model_name} | sed -s "s/\//_/g")

model_folder=models/curriculum_${EXP_ID}_${model_name_log}_${decoding_format}_${data_name}_${lr_scheduler}_lr${lr}_ls${label_smoothing}_${batch_size}_wu${warmup_steps}
data_folder=data/text2${decoding_format}/${data_name}

output_dir=${model_folder}
mkdir -p ${output_dir}

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python run_seq2seq.py \
    --do_train --do_eval --do_predict ${constraint_decoding} \
    --label_smoothing_sum=False \
    --use_fast_tokenizer=False \
    --evaluation_strategy steps \
    --predict_with_generate \
    --metric_for_best_model eval_trigger-F1 \
    --save_total_limit 1 \
    --load_best_model_at_end \
    --max_source_length=256 \
    --max_target_length=128 \
    --num_train_epochs=${pretrain} \
    --task=${task_name} \
    --train_file=${data_folder}_span/train.json \
    --validation_file=${data_folder}_span/val.json \
    --test_file=${data_folder}_span/test.json \
    --event_schema=${data_folder}_span/event.schema \
    --per_device_train_batch_size=${batch_size} \
    --per_device_eval_batch_size=$((batch_size * 4)) \
    --output_dir=${output_dir}/span_pretrain \
    --logging_dir=${output_dir}/span_pretrain_log \
    --model_name_or_path=${model_name} \
    --learning_rate=${lr} \
    --lr_scheduler_type=${lr_scheduler} \
    --label_smoothing_factor=${label_smoothing} \
    --eval_steps ${eval_steps} \
    --decoding_format treespan \
    --warmup_steps ${warmup_steps} \
    --source_prefix="span: " \
    --seed=${seed} --disable_tqdm True >${output_dir}/span_pretrain.log 2>${output_dir}/span_pretrain.log

mkdir ${output_dir}/span_pretrained
cp ${output_dir}/span_pretrain/config.json ${output_dir}/span_pretrained
cp ${output_dir}/span_pretrain/pytorch_model.bin ${output_dir}/span_pretrained
cp ${output_dir}/span_pretrain/special_tokens_map.json ${output_dir}/span_pretrained
cp ${output_dir}/span_pretrain/spiece.model ${output_dir}/span_pretrained
cp ${output_dir}/span_pretrain/tokenizer_config.json ${output_dir}/span_pretrained

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python run_seq2seq.py \
    --do_train --do_eval --do_predict ${constraint_decoding} \
    --label_smoothing_sum=False \
    --use_fast_tokenizer=False \
    --evaluation_strategy steps \
    --predict_with_generate \
    --metric_for_best_model eval_role-F1 \
    --save_total_limit 1 \
    --load_best_model_at_end \
    --max_source_length=256 \
    --max_target_length=128 \
    --num_train_epochs=${epoch} \
    --task=${task_name} \
    --train_file=${data_folder}/train.json \
    --validation_file=${data_folder}/val.json \
    --test_file=${data_folder}/test.json \
    --event_schema=${data_folder}/event.schema \
    --per_device_train_batch_size=${batch_size} \
    --per_device_eval_batch_size=$((batch_size * 4)) \
    --output_dir=${output_dir}/event_finetune \
    --logging_dir=${output_dir}/event_finetune_log \
    --model_name_or_path=${output_dir}/span_pretrained \
    --learning_rate=${lr} \
    --lr_scheduler_type=${lr_scheduler} \
    --label_smoothing_factor=${label_smoothing} \
    --eval_steps ${eval_steps} \
    --decoding_format ${decoding_format} \
    --warmup_steps ${warmup_steps} \
    --source_prefix="${task_name}: " \
    --seed=${seed} --disable_tqdm True >${output_dir}/event_finetune.log 2>${output_dir}/event_finetune.log


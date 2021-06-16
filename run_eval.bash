export device="0"
export model_path=""
export data_name=data/text2tree/one_ie_ace2005_subtype
export task_name="event"
export batch=1
export decoding_format='tree'

while getopts "b:d:m:i:t:co:f:" arg; do
  case $arg in
  b)
    batch=$OPTARG
    ;;
  d)
    device=$OPTARG
    ;;
  m)
    model_path=$OPTARG
    ;;
  i)
    data_folder=$OPTARG
    ;;
  t)
    task_name=$OPTARG
    ;;
  f)
    decoding_format=$OPTARG
    ;;
  c)
    constraint_decoding="--constraint_decoding"
    ;;
  o)
    extra_cmd=$OPTARG
    ;;
  ?)
    echo "unkonw argument"
    exit 1
    ;;
  esac
done

echo "Extra CMD: " "${extra_cmd}"

CUDA_VISIBLE_DEVICES=${device} python run_seq2seq.py \
  --do_eval --do_predict --task=${task_name} --predict_with_generate \
  --validation_file=${data_folder}/val.json \
  --test_file=${data_folder}/test.json \
  --event_schema=${data_folder}/event.schema \
  --model_name_or_path=${model_path} \
  --output_dir="${model_path}"_eval \
  --source_prefix="${task_name}: " \
  ${constraint_decoding} ${extra_cmd} \
  --per_device_eval_batch_size=${batch} \
  --decoding_format ${decoding_format}

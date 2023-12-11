device="0"
model_path=""
data_folder=data/text2spotasoc/absa/14lap
task_name="meta"
batch=16
decoding_format='spotasoc'
beam_size=1
map_config=config/offset_map/closest_offset_en.yaml

export PYTHONPATH="${PYTHONPATH}:./"

OPTS=$(getopt -o b:d:m:i:t:co:f:e: --long batch:,device:,model:,data:,task:constraint_decoding,output:,format:,map_config:,extra_cmd:, -n 'parse-options' -- "$@")

if [ $? != 0 ]; then
  echo "Failed parsing options." >&2
  exit 1
fi

eval set -- "$OPTS"

while true; do
  case "$1" in
    -b | --batch) batch="$2"
      shift 2 ;;
    -d | --device) device="$2"
      shift 2 ;;
    -m | --model) model_path="$2"
      shift 2 ;;
    -i | --data) data_folder="$2"
      shift 2 ;;
    -t | --task) task_name="$2"
      shift 2 ;;
    -c | --constraint_decoding) constraint_decoding="--constraint_decoding"
      shift ;;
    -o | --output) output_dir="$2"
      shift 2 ;;
    -f | --format) decoding_format="$2"
      shift 2 ;;
    -e | --extra_cmd) extra_cmd="$2"
      shift 2 ;;
    --beam) beam_size="$2"
      shift 2 ;;
    --map_config) map_config="$2"
      shift 2 ;;
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

echo "Extra CMD: " "${extra_cmd}"

if [[ ${output_dir} == "" ]]
then
  output_dir=${model_path}_eval
  if [[ ${constraint_decoding} != "" ]]
  then
    output_dir=${output_dir}_CD
  fi
fi

CUDA_VISIBLE_DEVICES=${device} python3 run_seq2seq.py \
  --use_fast_tokenizer=True \
  --max_source_length=${max_source_length:-"256"} \
  --max_target_length=${max_target_length:-"192"} \
  --do_eval --do_predict --task=record --predict_with_generate \
  --validation_file=${data_folder}/val.json \
  --test_file=${data_folder}/test.json \
  --record_schema=${data_folder}/record.schema \
  --model_name_or_path=${model_path} \
  --output_dir=${output_dir} \
  --source_prefix="${task_name}: " \
  --no_remove_unused_columns \
  --num_beams=${beam_size} \
  ${constraint_decoding} ${extra_cmd} \
  --per_device_eval_batch_size=${batch} \
  --decoding_format ${decoding_format}

python3 scripts/sel2record.py -p ${output_dir} -g ${data_folder} -v -d ${decoding_format} -c ${map_config}
python3 scripts/eval_extraction.py -p ${output_dir} -g ${data_folder} -w -m ${eval_match_mode:-"normal"}

echo "Output Dir:" ${output_dir}

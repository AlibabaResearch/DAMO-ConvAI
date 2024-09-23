
MODEL_CONFIG="1"
ARGS=(
  "--models" "$MODEL_CONFIG"
  "--eval_model" "$MODEL_CONFIG"
  "--debug_num" "$MODEL_CONFIG"
  "--doc_path" "$MODEL_CONFIG"
  "--input_path" "$MODEL_CONFIG"
  "--output_process_path" "$MODEL_CONFIG"
  "--output_path" "$MODEL_CONFIG"
  "--evaluate_output_path" "$MODEL_CONFIG"
  "--max_length" "$MODEL_CONFIG"
  "--model_config_dir" "$MODEL_CONFIG"
  "--process_num_gen" "$MODEL_CONFIG"
  "--process_num_eval" "$MODEL_CONFIG"
  "--rag"
  "--tmp" "$MODEL_CONFIG"
)

# Check whether the incoming parameters contain --continue_gen
for param in "$@"; do
  if [ "$param" == "--continue_gen" ]; then
    ARGS+=("--continue_gen")
  fi
done

echo "${ARGS[@]}"
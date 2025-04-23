## Agent vs Storyteller Scripts

### Basic Scripts
Here are some of the script for running {gpt-3.5-turbo, mixtral-7b-moe} under {normal interaction, omniscient interaction, script generation} mode in {normal, lite} setting.
If you need to run all interaction mode, you can use `run_all.sh`, the usage is `Usage: ./run_all.sh <model_name> <tag_base> <lite>`. For example, `./run_all.sh gpt-3.5-turbo exp0128 True`. You may find model_name in `LLM_Name`, and currently we are using `mistralai/Mixtral-8x7B-Instruct-v0.1` and `gpt-3.5-turbo`.
If you want to run mode separately, you can use `run_interaction.sh` or `run_script_full.sh`.
After running the above script, you may specify tags and fix those error episodes using `./fix_missing_episodes_with_tag.sh`.
Current `fix_missing_episodes_with_tag.py` first detects erroneous episodes, delete them and regenerate them.

### Fine-tuning

* `evaluate_finetuned_full.sh`: evaluate the fine-tuned model (gpt-3.5 finetuned on the full dataset) on the sotopia lite setting.
* `evaluate_finetuned_MF.sh`: evaluate the fine-tuned model (gpt-3.5 finetuned on the lite dataset) on the mutual friends setting.

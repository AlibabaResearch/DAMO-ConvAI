import os
import jsonlines
import pandas as pd
import argparse

def compute_session_metrics(input_directory, output_excel=''):
    final_progress = []
    final_all_session = 0
    final_right_session = 0
    final_right_api_num = 0
    final_all_api_num_gt = 0
    final_all_api_num_pre = 0 
    if output_excel:
        excel_path = pd.ExcelWriter(output_excel, engine='xlsxwriter')
    jsonl_files = [f for f in os.listdir(input_directory) if f.endswith('.jsonl')]

    for file_name in jsonl_files:
        file_path = os.path.join(input_directory, file_name)
        gpt_success = []
        gpt_progress = []
        api_num_right = []
        api_num_all_gt = []
        api_num_all_pre = []
        
        data = []
        with jsonlines.open(file_path) as reader:
            for obj in reader:
                data.append(obj)
                gpt_success.append(int(obj.get('success_gpt')))
                gpt_progress.append(float(obj.get('progress_gpt')))
                api_num_right.append(obj.get('right_api_num'))
                api_num_all_gt.append(obj.get('all_api_num_gt'))
                api_num_all_pre.append(obj.get('all_api_num_pre'))
        tmp_output = {
            "scenarios": file_name,
            "success_rate": sum(gpt_success) / len(gpt_success) if gpt_success else 0,
            "avg_progress": sum(gpt_progress) / len(gpt_progress) if gpt_progress else 0,
            "tool_precision": sum(api_num_right) / sum(api_num_all_pre) if api_num_all_pre else 0,
            "tool_recall": sum(api_num_right) / sum(api_num_all_gt) if api_num_all_gt else 0,
        }
        
        print(tmp_output)
        final_progress.extend(gpt_progress)
        final_all_session += len(gpt_success)
        final_right_session += sum(gpt_success)
        
        final_right_api_num += sum(api_num_right)
        final_all_api_num_gt += sum(api_num_all_gt)
        final_all_api_num_pre += sum(api_num_all_pre)
        if output_excel:
            df = pd.DataFrame(data)
            df.to_excel(excel_path, sheet_name=file_name.split('.')[0][:], index=False)
        
    final_gpt_success = final_right_session / final_all_session if final_all_session > 0 else 0
    final_gpt_progress = sum(final_progress) / len(final_progress) if final_progress else 0
    final_api_prec = final_right_api_num / final_all_api_num_pre if final_all_api_num_pre > 0 else 0
    final_api_recall = final_right_api_num / final_all_api_num_gt if final_all_api_num_gt > 0 else 0

    final_tmp_output = {
        "scenarios": "All",
        "success_rate": final_gpt_success,
        "avg_progress": final_gpt_progress,
        "tool_precision": final_api_prec,
        "tool_recall": final_api_recall
    }
    print("--------------")
    print(final_tmp_output)
    print(final_all_session)
    if output_excel:
        df_final = pd.DataFrame([final_tmp_output])
        df_final.to_excel(excel_path, sheet_name='Overall Metrics', index=False)
        excel_path._save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process paths and modes.")
    
    # Add arguments
    parser.add_argument("--output_excel", help="Path to the output Excel file")
    parser.add_argument("--eval_path", required=True, help="Path to the input directory for metric display")

    # Parse arguments
    args = parser.parse_args()
    ret = compute_session_metrics(args.eval_path,args.output_excel)
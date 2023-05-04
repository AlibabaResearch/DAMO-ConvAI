import json
import argparse
import pdb

def fetch_sql(predicted_results, output_path=None):
    final_sql = {}
    invalid_result = []
    for k, v in predicted_results.items():
        idx = int(k)
        print("------------------- processing {}th example -------------------".format(idx))
        print(v)
        try:
            cot, sql = v.split(': SELECT')
            clean_sql = 'SELECT' + sql
        except Exception as e:
            invalid_result.append(idx)
            clean_sql = 0 # filter resutls without valid SQL, i.e., too long, etc.
        final_sql[k] = clean_sql
    
    if output_path:
        json.dump(final_sql, open(output_path, 'w'), indent=4)
    return final_sql, invalid_result



if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--predicted_sql_path', type=str, required=True, default='')
    args_parser.add_argument('--output_clean_path', type=str, required=True, default='')
    args = args_parser.parse_args()
    exec_result = []
    
    # generate sql file:
    pred_file = json.load(open(args.predicted_sql_path, 'r'))
    post_sql, invalid_results = fetch_sql(pred_file, args.output_clean_path)
    
    print("filtered results, among {} examples, {} results are invaid".format(len(post_sql), len(invalid_results)))
    
# For sync the envs.
import random
import json
import pandas as pd
import pickle
from nsql.qa_module.openai_qa import OpenAIQAModel
import os
import time
from subprocess import PIPE, Popen
import uuid


# For Python execution.
class Executor(object):
    def __init__(self, args, keys=None):
        self.new_col_name_id = 0
        self.qa_model = OpenAIQAModel(args, keys)

    def nsql_exec(self, nsql: str, db: pd.DataFrame, verbose=True):
        # Add import part
        import_part = """import random
import json
import pandas as pd
import pickle
import numpy as np
from collections.abc import Iterable
from nsql.qa_module.openai_qa import OpenAIQAModel
from nsql.database import NeuralDB
import copy
import os
import time
verbose = {}""".format(str(verbose))

        # Add qa_map function
        qa_map_function_part = """def qa_map(db: pd.DataFrame, question, columns):
    new_db = NeuralDB([{"title": "", "table": {"header": db.columns.values.tolist(), "rows": db.values.tolist()}}])
    sql_executed_sub_tables = []
    for column in columns:
        column = f"`{column}`"
        sql_executed_sub_tables.append(new_db.execute_query(column))
        sub_table = qa_model.qa(question,
                                          sql_executed_sub_tables,
                                          table_title=new_db.table_title,
                                          qa_type="map",
                                          new_col_name_s=[question],
                                          verbose=verbose)
        new_db.add_sub_table(sub_table, verbose=verbose)
    table = new_db.get_table()
    return pd.DataFrame(table["rows"], columns=table["header"])"""

        # Add qa_ans function
        qa_ans_function_part = """def qa_ans(db: pd.DataFrame, question, columns):
    new_db = NeuralDB([{"title": "", "table": {"header": db.columns.values.tolist(), "rows": db.values.tolist()}}])
    sql_executed_sub_tables = []
    for column in columns:
        column = f"`{column}`"
        sql_executed_sub_tables.append(new_db.execute_query(column))
        answer = qa_model.qa(question,sql_executed_sub_tables,table_title=new_db.table_title,qa_type="ans",verbose=verbose)
    return answer"""

        # Convert np number type to python type
        convert_part = """def nested_to_python_number(x):
    if isinstance(x, np.int64):
        return int(x)
    if isinstance(x, np.float64):
        return float(x)
    if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
        return [nested_to_python_number(d) for d in x]
    return x"""
        # The prediction is a neural-python.

        # Add main function
        tmp_root_path = "tmp_python"
        os.makedirs(tmp_root_path, exist_ok=True)
        # Save the db
        db_file_path = '{}.db'.format(format(uuid.uuid4()))
        db_path = os.path.join(tmp_root_path, db_file_path)
        with open(db_path, "wb") as f:
            pickle.dump(db, f)

        # Save the qa_model
        model_file_path = '{}.model'.format(format(uuid.uuid4()))
        model_path = os.path.join(tmp_root_path, model_file_path)
        with open(model_path, "wb") as f:
            pickle.dump(self.qa_model, f)

        # Set the result path
        result_file_path = '{}.json'.format(format(uuid.uuid4()))
        result_path = os.path.join(tmp_root_path, result_file_path)

        # Read it and call solve function
        main_part = """if __name__ == '__main__':
    with open("{}", "rb") as f:
        db = pickle.load(f)
    with open("{}", "rb") as f:
        qa_model = pickle.load(f)
    result = solve(db)
    result = nested_to_python_number(result)
    with open("{}", "w") as f:
        json.dump(result, f)""".format(db_path, model_path, result_path)

        # Concat the code and execute the python
        all_code = "{}\n\n{}\n\n{}\n\n{}\n\n".format(import_part, qa_map_function_part, qa_ans_function_part,
                                                     convert_part) + nsql + "\n\n" + main_part

        if verbose:
            print("----> Code <----")
            print(all_code)

        python_file_path = '{}.py'.format(format(uuid.uuid4()))
        python_path = os.path.join(tmp_root_path, python_file_path)
        with open(python_path, "w") as f:
            f.write(all_code)

        p = Popen("python " + python_path, shell=True, stdout=PIPE, stderr=PIPE)
        stdout, stderr = p.communicate()

        # Error in execution so that we didn't get result.
        if not os.path.exists(result_path):
            print("stderr: ", stderr)
            raise ValueError("Error execution!")

        # Read the result
        with open(result_path, "r") as f:
            result = json.load(f)

        return result

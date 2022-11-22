import os, sys
import json
import sqlite3
import traceback
import argparse
import tqdm

from ..process_sql import get_sql
from .schema import Schema, get_schemas_from_json


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--tables", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    sql_path = args.input
    output_file = args.output
    table_file = args.tables

    schemas, db_names, tables = get_schemas_from_json(table_file)

    with open(sql_path) as inf:
        sql_data = json.load(inf)

    sql_data_new = []
    for data in tqdm.tqdm(sql_data):
        try:
            db_id = data["db_id"]
            schema = schemas[db_id]
            table = tables[db_id]
            schema = Schema(schema, table)
            sql = data["query"]
            sql_label = get_sql(schema, sql)
            data["sql"] = sql_label
            sql_data_new.append(data)
        except:
            print("db_id: ", db_id)
            print("sql: ", sql)
            raise

    with open(output_file, "wt") as out:
        json.dump(sql_data_new, out, sort_keys=True, indent=4, separators=(",", ": "))

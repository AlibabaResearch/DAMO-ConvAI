import argparse
import os

import sqlite3 as db
import pprint

def process_schema(args):
    schema_name = args.db
    schema_path = os.path.join(args.root_path, schema_name, schema_name + '.sqlite')
    print("load db data from", schema_path)
    conn = db.connect(schema_path)
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables_name = cur.fetchall()
    table_name_lst = [tuple[0] for tuple in tables_name]
    print(table_name_lst)
    for table_name in table_name_lst:
        cur.execute("SELECT * FROM %s" % table_name)
        col_name_list = [tuple[0] for tuple in cur.description]
        print(table_name, col_name_list)
        tables = cur.fetchall()
        #print(tables)


def main(args):
    process_schema(args)


    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, default="./data/database")
    parser.add_argument("--save_path", type=str, default="./database_content.json")
    parser.add_argument("--db", type=str, default="aircraft")
    args = parser.parse_args()
    main(args)

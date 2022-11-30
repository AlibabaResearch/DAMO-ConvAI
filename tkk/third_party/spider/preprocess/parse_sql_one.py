import os
import traceback
import re
import sys
import json
import sqlite3
import random
from os import listdir, makedirs
from collections import OrderedDict
from nltk import word_tokenize, tokenize
from os.path import isfile, isdir, join, split, exists, splitext

from ..process_sql import get_sql
from .schema import Schema, get_schemas_from_json


if __name__ == "__main__":

    sql = "SELECT name ,  country ,  age FROM singer ORDER BY age DESC"
    db_id = "concert_singer"
    table_file = "tables.json"

    schemas, db_names, tables = get_schemas_from_json(table_file)
    schema = schemas[db_id]
    table = tables[db_id]
    schema = Schema(schema, table)
    sql_label = get_sql(schema, sql)

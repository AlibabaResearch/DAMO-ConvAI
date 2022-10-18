## Data Preprocess

#### Get Parsed SQL Output

The SQL parsing script is `process_sql.py` in the main directory. Please refer to `parsed_sql_examples.sql` for the explanation of some parsed SQL output examples.

If you would like to use `process_sql.py` to parse SQL queries by yourself, `parse_sql_one.py` provides an example of how the script is called. Or you can use `parse_raw_json.py` to update all parsed SQL results (value for `sql`) in `train.json` and `dev.json`.

#### Get Table Info from Database

To generate the final `tables.json` file. It reads sqlite files from `database/` dir and previous `tables.json` with hand-corrected names: 

```
python process/get_tables.py [dir includes many subdirs containing database.sqlite files] [output file name e.g. output.json] [existing tables.json file to be inherited]
```

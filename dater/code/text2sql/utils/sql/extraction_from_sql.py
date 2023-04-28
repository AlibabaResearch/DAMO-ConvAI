import argparse
import json
from utils.sql.process_sql import (
  tokenize, CLAUSE_KEYWORDS, WHERE_OPS, COND_OPS, UNIT_OPS, AGG_OPS,
  JOIN_KEYWORDS, ORDER_OPS, skip_semicolon, SQL_OPS)
KEPT_WHERE_OP = ('not', 'in', 'exists')


def parse_table_unit(toks, start_idx, tables_with_alias):
  idx = start_idx
  len_ = len(toks)
  key = toks[idx]

  if idx + 1 < len_ and toks[idx + 1] == "as":
    tables_with_alias[toks[idx + 2]] = toks[idx]
    idx += 3
  else:
    idx += 1

  return idx, key

def parse_col(toks, start_idx, tables_with_alias, schema, default_tables=None):
  """
      :returns next idx, column id
  """
  tok = toks[start_idx]
  if tok == "*":
    return start_idx + 1

  if '.' in tok:  # if token is a composite
    alias, col = tok.split('.')
    # key = tables_with_alias[alias] + "." + col
    table = tables_with_alias[alias]
    """
    Add schema
    """
    if table not in schema:
      schema[table] = []
    schema[table].append(col)
    # We also want to normalize the column
    toks[start_idx] = "{}.{}".format(table, col)
    """
    END
    """
    return start_idx + 1

  assert default_tables is not None and len(default_tables) > 0, "Default tables should not be None or empty"

  # assert len(default_tables) == 1, "Default table should only have one time"

  """
  Add schema
  """
  # Find the best table here
  def choose_best_table(default_tables, tok):
    lower_tok = tok.lower()
    candidate = process.extractOne(lower_tok, [table.lower() for table in default_tables])[0]
    return candidate

  if len(default_tables) != 1:
    # print(default_tables)
    table = choose_best_table(default_tables, tok)
    # assert len(default_tables) == 1, "Default table should only have one time"
  else:
    table = default_tables[0]
  if table not in schema:
    schema[table] = []
  schema[table].append(tok)
  toks[start_idx] = "{}.{}".format(table, tok)
  return start_idx + 1

  # for alias in default_tables:
  #   table = tables_with_alias[alias]
  #   if tok in schema.schema[table]:
  #     key = table + "." + tok
  #     return start_idx + 1, schema.idMap[key]

  # assert False, "Error col: {}".format(tok)

def parse_col_unit(toks, start_idx, tables_with_alias, schema, default_tables=None, end_idx=None):
  """
      :returns next idx, (agg_op id, col_id)
  """
  idx = start_idx
  if end_idx is not None:
    len_ = len(toks[start_idx:end_idx])
  else:
    len_ = len(toks)
  isBlock = False
  isDistinct = False
  if toks[idx] == '(':
    isBlock = True
    idx += 1

  if toks[idx] in AGG_OPS:
    agg_id = AGG_OPS.index(toks[idx])
    idx += 1
    assert idx < len_ and toks[idx] == '('
    idx += 1
    if toks[idx] == "distinct":
      idx += 1
      isDistinct = True
    idx = parse_col(toks, idx, tables_with_alias, schema, default_tables)
    assert idx < len_ and toks[idx] == ')'
    idx += 1
    return idx

  if toks[idx] == "distinct":
    idx += 1
    isDistinct = True
  agg_id = AGG_OPS.index("none")
  idx = parse_col(toks, idx, tables_with_alias, schema, default_tables)

  if isBlock:
    assert toks[idx] == ')'
    idx += 1  # skip ')'

  return idx

def parse_val_unit(toks, start_idx, tables_with_alias, schema, default_tables=None):
  idx = start_idx
  len_ = len(toks)
  isBlock = False
  if toks[idx] == '(':
    isBlock = True
    idx += 1

  col_unit1 = None
  col_unit2 = None
  unit_op = UNIT_OPS.index('none')

  idx = parse_col_unit(toks, idx, tables_with_alias, schema, default_tables)
  if idx < len_ and toks[idx] in UNIT_OPS:
    unit_op = UNIT_OPS.index(toks[idx])
    idx += 1
    idx = parse_col_unit(toks, idx, tables_with_alias, schema, default_tables)

  if isBlock:
    assert toks[idx] == ')'
    idx += 1  # skip ')'

  return idx

def parse_value(toks, start_idx, tables_with_alias, schema, default_tables=None):
  idx = start_idx
  len_ = len(toks)

  isBlock = False
  if toks[idx] == '(':
    isBlock = True
    idx += 1

  if toks[idx] == 'select':
    idx = parse_sql(toks, idx, schema)
  elif "\"" in toks[idx]:  # token is a string value
    val = toks[idx]
    # Replace with placeholder
    toks[idx] = "_str_value_"
    idx += 1
  else:
    try:
      val = float(toks[idx])
      toks[idx] = "_num_value_"
      idx += 1
    except:
      end_idx = idx
      while end_idx < len_ and toks[end_idx] != ',' and toks[end_idx] != ')' \
              and toks[end_idx] != 'and' and toks[end_idx] not in CLAUSE_KEYWORDS and toks[
        end_idx] not in JOIN_KEYWORDS:
        end_idx += 1

      # idx = parse_col_unit(toks[start_idx: end_idx], 0, tables_with_alias, schema, default_tables)
      idx = parse_col_unit(toks, start_idx, tables_with_alias, schema, default_tables, end_idx=end_idx)
      idx = end_idx

  if isBlock:
    assert toks[idx] == ')'
    idx += 1

  return idx

def parse_condition(toks, start_idx, tables_with_alias, schema, default_tables=None):
  idx = start_idx
  len_ = len(toks)
  # conds = []

  while idx < len_:
    idx = parse_val_unit(toks, idx, tables_with_alias, schema, default_tables)
    not_op = False
    if toks[idx] == 'not':
      not_op = True
      idx += 1

    assert idx < len_ and toks[idx] in WHERE_OPS, "Error condition: idx: {}, tok: {}".format(idx, toks[idx])
    op_id = WHERE_OPS.index(toks[idx])
    idx += 1
    val1 = val2 = None
    if op_id == WHERE_OPS.index('between'):  # between..and... special case: dual values
      idx = parse_value(toks, idx, tables_with_alias, schema, default_tables)
      assert toks[idx] == 'and'
      idx += 1
      idx = parse_value(toks, idx, tables_with_alias, schema, default_tables)
    else:  # normal case: single value
      idx = parse_value(toks, idx, tables_with_alias, schema, default_tables)
      val2 = None

    # conds.append((not_op, op_id, val_unit, val1, val2))

    if idx < len_ and (toks[idx] in CLAUSE_KEYWORDS or toks[idx] in (")", ";") or toks[idx] in JOIN_KEYWORDS):
      break

    if idx < len_ and toks[idx] in COND_OPS:
      # conds.append(toks[idx])
      idx += 1  # skip and/or
  return idx# , conds


def parse_from(toks, start_idx, schema):
  assert 'from' in toks[start_idx:], "'from' not found"
  tables_with_alias = {}

  len_ = len(toks)
  idx = toks.index('from', start_idx) + 1
  default_tables = []
  table_units = []
  conds = []
  # print(idx, len_)
  while idx < len_:
    # print("idx", idx, toks[idx])
    isBlock = False
    if toks[idx] == '(':
      isBlock = True
      idx += 1

    if toks[idx] == 'select':
      idx = parse_sql(toks, idx, schema)
      # table_units.append((TABLE_TYPE['sql'], sql))
    else:
      if idx < len_ and toks[idx] == 'join':
        idx += 1  # skip join
      idx, table_name = parse_table_unit(toks, idx, tables_with_alias)
      # print(table_name)
      # table_units.append((TABLE_TYPE['table_unit'], table_unit))
      default_tables.append(table_name)
      """
      Add schema
      """
      if table_name not in schema:
        schema[table_name] = []
      """
      END
      """

    if idx < len_ and toks[idx] == "on":
      idx += 1  # skip on
      idx = parse_condition(toks, idx, tables_with_alias, schema, default_tables)
      # if len(conds) > 0:
      #   conds.append('and')
      # conds.extend(this_conds)

    if isBlock:
      assert toks[idx] == ')'
      idx += 1

    if idx < len_ and (toks[idx] in CLAUSE_KEYWORDS or toks[idx] in (")", ";")):
      break

  return idx, default_tables, tables_with_alias

def parse_select(toks, start_idx, tables_with_alias, schema, default_tables=None):
  idx = start_idx
  len_ = len(toks)

  assert toks[idx] == 'select', "'select' not found"
  idx += 1
  isDistinct = False
  if idx < len_ and toks[idx] == 'distinct':
    idx += 1
    isDistinct = True
  val_units = []

  while idx < len_ and toks[idx] not in CLAUSE_KEYWORDS:
    agg_id = AGG_OPS.index("none")
    if toks[idx] in AGG_OPS:
      agg_id = AGG_OPS.index(toks[idx])
      idx += 1
    idx = parse_val_unit(toks, idx, tables_with_alias, schema, default_tables)
    # val_units.append((agg_id, val_unit))
    if idx < len_ and toks[idx] == ',':
      idx += 1  # skip ','

  return idx

def parse_where(toks, start_idx, tables_with_alias, schema, default_tables):
  idx = start_idx
  len_ = len(toks)

  if idx >= len_ or toks[idx] != 'where':
    return idx

  idx += 1
  idx = parse_condition(toks, idx, tables_with_alias, schema, default_tables)
  return idx

def parse_group_by(toks, start_idx, tables_with_alias, schema, default_tables):
  idx = start_idx
  len_ = len(toks)
  col_units = []

  if idx >= len_ or toks[idx] != 'group':
    return idx

  idx += 1
  assert toks[idx] == 'by'
  idx += 1

  while idx < len_ and not (toks[idx] in CLAUSE_KEYWORDS or toks[idx] in (")", ";")):
    idx = parse_col_unit(toks, idx, tables_with_alias, schema, default_tables)
    # col_units.append(col_unit)
    if idx < len_ and toks[idx] == ',':
      idx += 1  # skip ','
    else:
      break

  return idx

def parse_having(toks, start_idx, tables_with_alias, schema, default_tables):
  idx = start_idx
  len_ = len(toks)

  if idx >= len_ or toks[idx] != 'having':
    return idx

  idx += 1
  idx = parse_condition(toks, idx, tables_with_alias, schema, default_tables)
  return idx

def parse_order_by(toks, start_idx, tables_with_alias, schema, default_tables):
  idx = start_idx
  len_ = len(toks)
  val_units = []
  order_type = 'asc'  # default type is 'asc'

  if idx >= len_ or toks[idx] != 'order':
    return idx

  idx += 1
  assert toks[idx] == 'by'
  idx += 1

  while idx < len_ and not (toks[idx] in CLAUSE_KEYWORDS or toks[idx] in (")", ";")):
    idx = parse_val_unit(toks, idx, tables_with_alias, schema, default_tables)
    # val_units.append(val_unit)
    if idx < len_ and toks[idx] in ORDER_OPS:
      order_type = toks[idx]
      idx += 1
    if idx < len_ and toks[idx] == ',':
      idx += 1  # skip ','
    else:
      break

  return idx

def parse_limit(toks, start_idx):
  idx = start_idx
  len_ = len(toks)

  if idx < len_ and toks[idx] == 'limit':
    idx += 2
    toks[idx - 1] = "_limit_value_"
    # make limit value can work, cannot assume put 1 as a fake limit number
    if type(toks[idx - 1]) != int:
      return idx

    return idx

  return idx

def parse_sql(toks, start_idx, schema):
  isBlock = False  # indicate whether this is a block of sql/sub-sql
  len_ = len(toks)
  idx = start_idx

  if toks[idx] == '(':
    isBlock = True
    idx += 1

  from_end_idx, default_tables, tables_with_alias = parse_from(toks, start_idx, schema)

  _ = parse_select(toks, idx, tables_with_alias, schema, default_tables)
  idx = from_end_idx

  idx = parse_where(toks, idx, tables_with_alias, schema, default_tables)
  idx = parse_group_by(toks, idx, tables_with_alias, schema, default_tables)
  idx = parse_having(toks, idx, tables_with_alias, schema, default_tables)
  idx = parse_order_by(toks, idx, tables_with_alias, schema, default_tables)
  idx = parse_limit(toks, idx)
  #
  idx = skip_semicolon(toks, idx)
  if isBlock:
    assert toks[idx] == ')'
    idx += 1  # skip ')'
  idx = skip_semicolon(toks, idx)

  # for op in SQL_OPS:  # initialize IUE
  #   sql[op] = None
  if idx < len_ and toks[idx] in SQL_OPS:
    sql_op = toks[idx]
    idx += 1
    idx = parse_sql(toks, idx, schema)
    # sql[sql_op] = IUE_sql
  return idx

def extract_schema_from_sql(schema, sql):
  toks = tokenize(sql)
  parse_sql(toks=toks, start_idx=0, schema=schema)
  return toks

def extract_template_from_sql(sql, schema={}):
  try:
    toks = tokenize(sql)
  except:
    print("Tokenization error for {}".format(sql))
    toks = []
  # print(toks)
  template = []
  # ignore_follow_up_and = False
  len_ = len(toks)
  idx = 0
  while idx < len_:
    tok = toks[idx]
    if tok == "from":
      template.append(tok)
      if toks[idx+1] != "(":
        template.append("[FROM_PART]")
        idx += 1
        while idx < len_ and (toks[idx] not in CLAUSE_KEYWORDS and toks[idx] != ")"):
          idx += 1
        continue
    elif tok in CLAUSE_KEYWORDS:
      template.append(tok)
    elif tok in AGG_OPS:
      template.append(tok)
    elif tok in [",", "*", "(", ")", "having", "by", "distinct"]:
      template.append(tok)
    elif tok in ["asc", "desc"]:
      template.append("[ORDER_DIRECTION]")
    elif tok in WHERE_OPS:
      if tok in KEPT_WHERE_OP:
        template.append(tok)
      else:
        template.append("[WHERE_OP]")
        if tok == "between":
          idx += 2
    elif tok in COND_OPS:
      template.append(tok)
    elif template[-1] == "[WHERE_OP]":
      template.append("[VALUE]")
    elif template[-1] == "limit":
      template.append("[LIMIT_VALUE]")
    elif template[-1] != "[MASK]": # value, schema, join on as
      template.append("[MASK]")
    idx += 1
  return template

def extract_partial_template_from_sql(sql, schema={}):
  toks = tokenize(sql)
  # print(toks)
  template = []
  # ignore_follow_up_and = False
  len_ = len(toks)
  idx = 0
  while idx < len_:
    tok = toks[idx]
    if tok == "from":
      template.append(tok)
      if toks[idx+1] != "(":
        # template.append("[FROM_PART]")
        idx += 1
        while idx < len_ and (toks[idx] not in CLAUSE_KEYWORDS and toks[idx] != ")"):
          template.append(toks[idx])
          idx += 1
        continue
    elif tok in CLAUSE_KEYWORDS:
      template.append(tok)
    elif tok in AGG_OPS:
      template.append(tok)
    elif tok in [",", "*", "(", ")", "having", "by", "distinct"]:
      template.append(tok)
    elif tok in ["asc", "desc"]:
      template.append("[ORDER_DIRECTION]")
    elif tok in WHERE_OPS:
      if tok in KEPT_WHERE_OP:
        template.append(tok)
      else:
        template.append("[WHERE_OP]")
        if tok == "between":
          idx += 2
    elif tok in COND_OPS:
      template.append(tok)
    elif template[-1] == "[WHERE_OP]":
      template.append("[VALUE]")
    elif template[-1] == "limit":
      template.append("[LIMIT_VALUE]")
    else:
      template.append(tok)
    idx += 1
  return template


def is_valid_schema(schema):
  # There is no "." and " " in the column name
  for table in schema:
    if "." in table:
      return False
    if any([keyword == table for keyword in CLAUSE_KEYWORDS]):
      return False
    for column in schema[table]:
      if "." in column or " " in column or '"' in column or "'" in column:
        return False
  return True

def clean_sql(sql):
  while "JOIN JOIN" in sql:
    sql = sql.replace("JOIN JOIN", "JOIN")
  if "JOIN WHERE" in sql:
    sql = sql.replace("JOIN WHERE", "WHERE")
  if "JOIN GROUP BY" in sql:
    sql = sql.replace("JOIN GROUP BY", "GROUP BY")
  return sql

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--input_file", type=str)
  parser.add_argument("--output_file", type=str)
  parser.add_argument("--mode", type=str, choices=["debug", "verbose", "silent"])
  parser.add_argument("--task", type=str, choices=["template_extraction", "schema_extraction"])
  args = parser.parse_args()

  if args.task == "schema_extraction":
    if args.mode == "debug":
      sql = "SELECT count(*) FROM games"
      sql = sql + " INTERSECT " + "SELECT sacks, year FROM players"
      sql = sql + " EXCEPT " + 'SELECT T1.year, T1.sacks FROM players AS T1 JOIN tackles AS T2 ON T1.id = T2.player_id WHERE T2.manager = "A" and T2.season NOT IN (SELECT season FROM match WHERE match_name = "IVL" INTERSECT SELECT T1.year, T1.sacks FROM sack AS T1) GROUP BY T1.year, T1.sacks HAVING count(T1.coach) > 10 ORDER BY T2.score LIMIT 5'
      sql = "SELECT T1.pld FROM pld AS T1 JOIN games AS T2 ON T1.crs_code = T2.crs_code JOIN GROUP BY T1.pld WHERE T2.gf = '8' AND T2.gf = '9'"
      sql = 'select * from head where height = "6-0" or height = "6-0" order by height asc'
      schema = {}
      extract_schema_from_sql(schema, sql)
      print(schema, is_valid_schema(schema))
    elif args.mode == "verbose":
      fout = open(args.output_file, "w")
      with open(args.input_file) as fin:
        for line in fin:
          example = json.loads(line)
          schema = {}
          try:
            sql = example["sql"] if "sql" in example else example["pred"]
            sql = clean_sql(sql)
            example["sql"] = sql
            extract_schema_from_sql(schema, sql)

          except:
            # print(sql)
            continue
          for table in schema:
            schema[table] = list(set(schema[table]))
          if is_valid_schema(schema):
            example["extracted_schema"] = schema
            fout.write(json.dumps(example) + "\n")
    elif args.mode == "verbose":
      fout = open(args.output_file, "w")
      with open(args.input_file) as fin:
        for line in fin:
          example = json.loads(line)
          schema = {}
          sql = example["sql"] if "sql" in example else example["pred"]
          sql = clean_sql(sql)
          example["sql"] = sql
          extract_schema_from_sql(schema, sql)
          for table in schema:
            schema[table] = list(set(schema[table]))
          example["extracted_schema"] = schema
          fout.write(json.dumps(example) + "\n")
          if is_valid_schema(schema):
            example["extracted_schema"] = schema
            fout.write(json.dumps(example) + "\n")
  elif args.task == "template_extraction":
    if args.mode == "debug":
      sql = "SELECT avg(T1.Votes) FROM seats AS T1 JOIN votes AS T2 ON T1.Seat_ID = T2.Seat_ID WHERE T1.seats BETWEEN 1 AND 2 and T1.Seats = 1 AND T2.Votes = 10"
      print(extract_template_from_sql(sql))
      print(extract_partial_template_from_sql(sql))
    elif args.mode == "verbose":
      fout_json = open(args.output_file + ".json", "w")
      fout_txt = open(args.output_file + ".txt", "w")
      low_freq_txt = open(args.output_file + ".low_freq", "w")
      high_freq_txt = open(args.output_file + ".high_freq", "w")
      all_templates = set()
      # for input_file in args.input_file.split(","):
      templates = {}
      with open(args.input_file) as fin:
        for line in fin:
          example = json.loads(line)
          sql = example["sql"] if "sql" in example else example["pred"]
          if isinstance(sql, list):
            sql = sql[-1]
          template = extract_template_from_sql(sql)
          template_str = " ".join(template)
          if template_str not in templates:
            templates[template_str] = []
          templates[template_str].append(sql)
      print("{} has template {}".format(args.input_file, len(templates)))

      json.dump(templates, fout_json)
      for template in sorted(templates.keys()):
        if len(templates[template]) > 1:
          high_freq_txt.write(template + "\n")
        else:
          low_freq_txt.write(template + "\n")
        fout_txt.write(template + "\n")




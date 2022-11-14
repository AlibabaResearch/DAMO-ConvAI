# coding: utf-8

# In[1]:


import os
import re
import json
import pickle
import random
from template_config import *
from collections import defaultdict
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

import nltk

nltk.download('wordnet')
ps = PorterStemmer()
lmtzr = WordNetLemmatizer()

# In[3]:


def read_in_all_data(data_path=DATA_PATH):
    training_data = json.load(open(os.path.join(data_path, "train.json")))
    dev_data = json.load(open(os.path.join(data_path, "dev.json")))
    tables_org = json.load(open(os.path.join(data_path, "tables.json")))
    tables = {tab['db_id']: tab for tab in tables_org}

    return training_data, dev_data, tables


# In[4]:


def get_all_question_query_pairs(data):
    question_query_pairs = []
    for item in data:
        question_query_pairs.append((item['question_toks'], item['query'], item['db_id']))
    # question query pairs: [([question_tok], query(str), db_id(str)]
    return question_query_pairs


# In[7]:


def is_value(token):
    """
    as values can either be a numerical digit or a string literal, then we can
    detect if a token is a value by matching with regex
    """
    is_number = True
    try:
        float(token)
    except ValueError:
        is_number = False
    is_string = token.startswith("\"") or token.startswith("\'") or token.endswith("\"") or token.endswith("\'")

    return is_number or is_string


def remove_all_from_clauses(query_keywords):
    """
    remove all keywords from from clauses, until there is no more from clauses
    e.g. select {} from {} as {} where {} = {} --> select {} where {} = {}
    """
    # remove from clause by deleting the range from "FROM" to "WHERE" or "GROUP"
    table_map = defaultdict(list)
    count = -1
    while "FROM" in query_keywords:
        count += 1
        if count > MAX_FROM_COUNT:
            break
            print("error query_keywords: ", query_keywords)
        start_location = query_keywords.index("FROM")
        end_token_locations = [len(query_keywords)]  # defaulting to the end of the list
        for end_token in ["WHERE", "GROUP", "ORDER"]:
            try:
                end_token_locations.append(query_keywords.index(end_token, start_location))
            except ValueError:
                pass
        end_location = min(end_token_locations)
        # should not delete too much ')'
        nested_layer = 0
        for i, token in enumerate(query_keywords[start_location:min(end_token_locations)]):
            if token == '(':
                nested_layer += 1
            elif token == ')':
                nested_layer -= 1
            if nested_layer == -1:
                end_location = i + start_location
                break
        table_map['{SELECT' + str(count) + '}'] = [x for x in query_keywords[start_location:end_location] if '{TABLE' in x]
        query_keywords = query_keywords[:start_location] + [FROM_SYMBOL] + query_keywords[end_location:]

    return query_keywords, table_map


def remove_all_select_clauses(query_keywords):
    """
    remove all keywords from from clauses, until there is no more from clauses
    e.g. select {} from {} as {} where {} = {} --> select {} where {} = {}
    """
    # remove from clause by deleting the range from "FROM" to "WHERE" or "GROUP"
    select_clauses = []
    count = 0
    while "SELECT" in query_keywords:
        count += 1
        if count > MAX_SELECT_COUNT:
            break
            print("error query_keywords: ", query_keywords)
        start_location = query_keywords.index("SELECT")
        end_token_locations = [len(query_keywords)]  # defaulting to the end of the list
        for end_token in ["FROM", "{FROM}"]:
            try:
                end_token_locations.append(query_keywords.index(end_token, start_location))
            except ValueError:
                pass
        select_clauses.append(query_keywords[start_location:min(end_token_locations)])
        query_keywords = query_keywords[:start_location] + ["{SELECT}"] + query_keywords[min(end_token_locations):]

    return query_keywords, select_clauses


def clean_query(query):
    # clean query: replace values, numbers, column names with SYMBOL
    query = query.replace(";", "")
    query = query.replace("\t", "")
    query = query.replace("(", " ( ").replace(")", " ) ")
    return query


def extract_value_and_tokenize(query):
    value_dict = {}

    # replace all stuff enclosed by "" with a numerical value to get it marked as {VALUE}
    str_1 = re.findall("\"[^\"]*\"", query)
    str_2 = re.findall("\'[^\']*\'", query)
    values = str_1 + str_2
    for val in values:  # maybe there are no space between symbols and texts
        symbol = "{{VALUE{}}}".format(len(value_dict))
        query = query.replace(val.strip(), symbol)
        value_dict[symbol] = val.strip()

    # deal with numbers
    query_tokenized = query.split(' ')
    nums = re.findall("[-+]?\d*\.\d+|[-+]?\d+", query)
    query_tokenized = ["{{VALUE{}}}".format(len(value_dict)+nums.index(qt))
                       if qt in nums else qt for qt in query_tokenized]
    for num in nums:
        symbol = "{{VALUE{}}}".format(len(value_dict))
        value_dict[symbol] = num.strip()

    return query_tokenized, value_dict


def extract_table_names(query_tokenized, table):
    # get table column names info
    column_types = table['column_types']
    table_names_original = [cn.lower() for cn in table['table_names_original']]
    table_names = [cn.lower() for cn in table['table_names']]
    column_names = [cn.lower() for i, cn in table['column_names']]
    column_names_original = [cn.lower() for i, cn in table['column_names_original']]
    columns = table_names_original + table_names
    # deal with columns
    # query_tokenized = query.split(' ')
    cols_dict = {}
    column_dict = {}
    column_nl_dict = {}
    table_dict = {}
    query_keywords = []  # query tokens
    for token in query_tokenized:
        if len(token.strip()) == 0:  # in case there are more than one space used
            continue
        if IGNORE_COMMAS_AND_ROUND_BRACKETS:
            keywords_dict = SQL_KEYWORDS_AND_OPERATORS_WITHOUT_COMMAS_AND_BRACES
        else:
            keywords_dict = SQL_KEYWORDS_AND_OPERATORS
        # not a keyword and not a {value}, it is a name
        if token.upper() not in keywords_dict and token[0] != '{':
            token = token.upper()
            if USE_COLUMN_AND_VALUE_REPLACEMENT_TOKEN:
                token = re.sub("[T]\d+\.", '', token)
                token = re.sub(r"\"|\'", '', token)
                token = re.sub("[T]\d+", '', token).lower() # TODO: meaning of these three lines

                # Table_name.column_name
                if '.' in token:
                    token = token.split('.')[-1].strip()

                if token != '' and token in column_names_original:
                    try:
                        tok_ind = column_names_original.index(token)
                    except:
                        print("\ntable: {}".format(table['db_id']))
                        print("\ntoken: {}".format(token))
                        print("column_names_original: {}".format(column_names_original))
                        # print("query: {}".format(query))
                        print("query_tokenized: {}".format(query_tokenized))
                        exit(1)
                    col_type = column_types[tok_ind]
                    col_name = column_names[tok_ind]
                    columns.append(col_name)
                    columns.append(token)
                    if token not in cols_dict:
                        cols_dict[token] = COLUMN_SYMBOL.replace("}", str(len(cols_dict))+'}')
                        column_dict[cols_dict[token]] = token
                        column_nl_dict[cols_dict[token]] = col_name
                    query_keywords.append(cols_dict[token])
                elif token in table_names_original:
                    tok_ind = table_names_original.index(token)
                    table_name = table_names[tok_ind]

                    if table_name not in table_dict.values():
                        symbol = TABLE_SYMBOL[:-1] + str(len(table_dict)) + TABLE_SYMBOL[-1]
                        table_dict[symbol] = table_name
                    else:
                        for sym, token in table_dict.items():
                            if token == table_name:
                                symbol = sym
                                break
                    query_keywords.append(symbol)

        else:
            query_keywords.append(token.upper())
    return query_keywords, column_dict, table_dict, columns, column_nl_dict


def extract_sql_components(query_keywords, sql_components, type):
    compnent_list = sql_components[type]
    compnent_dict = {}
    query_new = []
    for token in query_keywords:
        if token.upper() in compnent_list:
            sym = "{{{}{}}}".format(type, len(compnent_dict))
            query_new.append(sym)
            compnent_dict[sym] = token.upper()
        else:
            query_new.append(token)
    return query_new, compnent_dict


def strip_query_full_dict(query, table):

    query = clean_query(query)

    query_tokenized, value_dict = extract_value_and_tokenize(query)
    query_keywords, column_dict, table_dict, columns, column_nl_dict = extract_table_names(query_tokenized, table)

    if "FROM" in query_keywords:
        query_keywords, table_map = remove_all_from_clauses(query_keywords)

    # load keys
    sql_components = json.load(open(SQL_COMPONENTS_PATH))
    # extract ops
    query_keywords, op_dict = extract_sql_components(query_keywords, sql_components, 'OP')
    query_keywords, agg_dict = extract_sql_components(query_keywords, sql_components, 'AGG')
    query_keywords, sc_dict = extract_sql_components(query_keywords, sql_components, 'SC')
    sql_component_dict = {**op_dict, **agg_dict, **sc_dict}

    if "SELECT" in query_keywords:
        query_keywords, select_clauses = remove_all_select_clauses(query_keywords)

    if USE_LIMITED_KEYWORD_SET:
        query_keywords = [kw for kw in query_keywords if kw in LIMITED_KEYWORD_SET]

    columns_lemed = [lmtzr.lemmatize(w) for w in " ".join(columns).split(" ") if w not in LOW_CHAR]
    columns_lemed_stemed = [ps.stem(w) for w in columns_lemed]

    return " ".join(query_keywords), value_dict, column_dict, table_dict, sql_component_dict, \
           columns_lemed_stemed, select_clauses, column_nl_dict, table_map


def process_question_full_dict(question, value_dict, column_dict, table_dict,
                               sql_component_dict, columns_lemed_stemed, select_clauses, column_nl_dict, table_map):
    question = " ".join(question).lower()
    # replace value
    value_dict = {sym: re.sub(r"\"|\'", '', val) for sym, val in value_dict.items()}
    for sym, val in value_dict.items():
        val = val.strip().lower()
        try:
            question = re.sub(r'\b' + val + r'\b', sym, question)
        except:
            print(sym, val, 'is not match')
            continue

    # replace columns and tables
    column_dict = {x: lmtzr.lemmatize(y) for x, y in column_dict.items() if y not in LOW_CHAR}
    column_dict = {x: ps.stem(y) for x, y in column_dict.items()}
    question_toks = question.split(" ")
    question_lemed = [lmtzr.lemmatize(w) for w in question_toks]
    question_lemed_stemed = [ps.stem(w) for w in question_lemed]
    replace_inds = [i for i, qt in enumerate(question_lemed_stemed) if qt in columns_lemed_stemed]

    for ind in replace_inds:
        found = 0
        column_token = question_toks[ind]
        for sym, token in column_dict.items():
            if token in column_token or column_token == token.split("_")[0]:
                question_toks[ind] = sym
            elif column_token in token.split('_'):
                found = 1
        if question_toks[ind][0] == '{':
            continue
        for sym, token in table_dict.items():
            if token in column_token or column_token == token.split("_")[0]:
                question_toks[ind] = sym
            elif column_token in token.split('_'):
                found = 1
        if question_toks[ind][0] != '{' and not found:
            question_toks[ind] = question_toks[ind]  # TODO
    question_template = ' '.join(question_toks)

    # replace part of sql components TODO

    return question_template


def filter_string(cs):
    return "".join([c.upper() for c in cs if c.isalpha() or c == ' '])


def general_pattern(pattern):
    general_pattern_list = []
    for x in pattern.split(" "):
        if x in KEY_KEYWORD_SET:
            general_pattern_list.append(x)

    return " ".join(general_pattern_list)


def sub_pattern(pattern):
    general_pattern_list = []
    for x in pattern.split(" "):
        if x in ALL_KEYWORD_SET:
            general_pattern_list.append(x)

    return " ".join(general_pattern_list)


def tune_pattern_with_index(pattern):
    general_pattern_list = []
    for x in pattern.split(" "):
        if "{COLUMN" in x:
            general_pattern_list.append(x + "}")
            continue

    return " ".join(general_pattern_list)


def tune_pattern(pattern, omit_distinct=True):
    general_pattern_list = []
    col, val, op, sc, agg, sel = [0]*6
    for x in pattern.split(" "):
        if "{SELECT" in x:
            general_pattern_list.append("{SELECT" + str(sel) + '}')
            sel += 1
        elif "{COLUMN" in x:
            general_pattern_list.append(COLUMN_SYMBOL.replace("}", str(col) + "}"))
            col += 1
        elif "{VALUE" in x:
            general_pattern_list.append("{VALUE" + str(val) + "}")
            val += 1
        elif x == 'DISTINCT' and omit_distinct:
            continue
        elif "{SC" in x:
            general_pattern_list.append("{SC"+str(sc) + "}")
            sc += 1
        elif "{OP" in x:
            general_pattern_list.append("{OP" + str(op) + "}")
            op += 1
        elif "{AGG" in x:
            general_pattern_list.append("{AGG" + str(agg) + "}")
            agg += 1
        else:
            general_pattern_list.append(x)

    return " ".join(general_pattern_list)


def get_pattern_question(train_qq_pairs, tables):
    pattern_question_dict = defaultdict(list)
    detailed_pattern_question_dict = defaultdict(list)

    # train_qq_pairs
    for eid, (question, query, bd_id) in enumerate(train_qq_pairs):
        table = tables[bd_id]
        if eid % 500 == 0:
            print("processing eid: ", eid)

        # # for debuging
        # if ' '.join(question) != 'Find the number of followers for each user .':
        #     continue
        pattern, *dicts = strip_query_full_dict(query, table)

        question_template = process_question_full_dict(question, *dicts)
        name_dicts = {**dicts[0], **dicts[1], **dicts[2], **dicts[3]}

        gen_pattern = general_pattern(pattern)
        more_pattern = sub_pattern(pattern)
        tu_pattern = tune_pattern(pattern)
        # tu_pattern = tune_pattern(pattern[pattern.index("WHERE"):] if "WHERE" in pattern else pattern)

        pattern_question_dict[tu_pattern].append(' '.join(question) + " ||| " + query + " ||| " +
                                                  question_template + " ||| " + more_pattern
                                                  + " ||| " + query)
        detailed_pattern_question_dict[tu_pattern].append(
            {
                'question': ' '.join(question),
                'query': query,
                'template': question_template,
                'concise pattern': more_pattern,
                'name dict': name_dicts
            }
        )

    #     print("\n--------------------------------------")
    #     print("original question: {}".format(' '.join(question).encode('utf-8')))
    #     print("question: {}".format(question_template.encode('utf-8')))
    #     print("query: {}".format(query))
    #     print("pattern: {}".format(pattern))
    #     print("values: {}".format(values))
    #     print("nums: {}".format(nums))
    #     print("columns: {}".format(columns))

    # In[10]:

    print("total pattern number: {}".format(len(pattern_question_dict)))
    pattern_question_dict = sorted(pattern_question_dict.items(), key=lambda kv: len(kv[1]), reverse=True)

    # [(pattern(str), [question(str)])]
    detailed_pattern_question_dict = sorted(detailed_pattern_question_dict.items(),
                                            key=lambda kv: (len(kv[1]), kv[0]), reverse=True)
    detailed_pattern_question_dict = [(x, sorted(y, key=lambda z: -z['template'].count('{')))
                                       for x, y in detailed_pattern_question_dict]

    return pattern_question_dict, detailed_pattern_question_dict


def clean_select(clause, table_dict):

    clause = [x[:-1]+"OLD}" if x[-1] == '}' else x+'OLD}'
              for x in clause if 'AGG' in x or 'COLUMN' in x]
    clause = ' , '.join(clause).split(' ')
    clause = [x for i, x in enumerate(clause)
              if x != ',' or not i or 'COLUMN' in clause[i-1]]

    clause += ('of '+' , '.join(table_dict[:MAX_TABLE_USED])).split(' ')
    return clause





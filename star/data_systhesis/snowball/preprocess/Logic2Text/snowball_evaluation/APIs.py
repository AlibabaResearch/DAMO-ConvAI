import numpy
import re
import math
import pandas as pd
import numpy as np
import datetime

APIs = {}

# With only one argument
min_dict = ['minimum', 'fastest', 'smallest', 'shortest', 'worst', 'youngest', 'least', 'lowest', 'earliest', 'any', 'highest rank', 'latest', 'longest', 'oldest']
max_dict = ['max', 'slowest', 'maximum', 'largest', 'longest', 'oldest', 'top', 'best', 'most', 'highest', 'latest', 'larger than any', 'lowest rank','predominantly', 'earliest', 'shortest', 'newest']

greater_dict = ['faster','greater', 'larger', 'older', 'more', 'older', 'bigger', 'larger', 'higher', 'more', 'after', 'greater', 'above', 'over', 'at least', 'worse rank', 'later', 'longer']
less_dict = ['slower','younger', 'smaller', 'less', 'fewer', 'younger', 'smaller', 'lower', 'less', 'before', 'shorter', 'below', 'under', 'lighter', "better rank", 'earlier', 'shorter', 'prior']
most_dict = ['majority', 'most', 'major']
### count
APIs['count'] = {"argument": ['row'], 'output': 'num',
                 'function': lambda t: len(t),
                 'tostr': lambda t: "count {{ {} }}".format(t),
                 'to_nl': lambda t: "the number of ( {} )".format(t),
                 'append': True}

### unique
APIs['only'] = {"argument": ['row'], 'output': 'bool',
                "function": lambda t: len(t) == 1,
                "tostr": lambda t: "only {{ {} }}".format(t),
                "to_nl": lambda t: "( {} ) only contains one value".format(t),
                'alias': ['only', 'one'],
                'append': None}

# With only two argument and the first is row
APIs['str_hop'] = {"argument": ['row', 'header'], 'output': 'str',
                   'function': lambda t, col: hop_op(t, col),
                   'tostr': lambda t, col: "hop {{ {} ; {} }}".format(t, col),
                   'to_nl': lambda t, col: "( {} ) of ( {} )".format(col, t),
                   'append': True}

APIs['num_hop'] = {"argument": ['row', 'header'], 'output': 'obj',
                   'function': lambda t, col: hop_op(t, col),
                   'tostr': lambda t, col: "hop {{ {} ; {} }}".format(t, col),
                   'to_nl': lambda t, col: "( {} ) of ( {} )".format(col, t),
                   'append': True}

APIs['avg'] = {"argument": ['row', 'header'], 'output': 'num',
               "function": lambda t, col: agg(t, col, "mean"),
               "tostr": lambda t, col: "avg {{ {} ; {} }}".format(t, col),
               "to_nl": lambda t, col: "average value of ( {} ) of ( {} )".format(col, t),  # TODO
               'alias': ['average', 'mean'],
               'append': True}

APIs['sum'] = {"argument": ['row', 'header'], 'output': 'num',
               "function": lambda t, col: agg(t, col, "sum"),
               "tostr": lambda t, col: "sum {{ {} ; {} }}".format(t, col),
               "to_nl": lambda t, col: "sum of ( {} ) of ( {} )".format(col, t),
               'alias': ['sum', 'total'],
               'append': True}

APIs['max'] = {"argument": ['row', 'header'], 'output': 'obj',
               "function": lambda t, col: nth_maxmin(t, col, order=1, max_or_min="max", arg=False),
               "tostr": lambda t, col: "max {{ {} ; {} }}".format(t, col),
               "to_nl": lambda t, col: "maximum of ( {} ) of ( {} )".format(col, t),
               'alias': max_dict,
               'append': True}

APIs['min'] = {"argument": ['row', 'header'], 'output': 'obj',
               "function": lambda t, col: nth_maxmin(t, col, order=1, max_or_min="min", arg=False),
               "tostr": lambda t, col: "min {{ {} ; {} }}".format(t, col),
               "to_nl": lambda t, col: "minimum of ( {} ) of ( {} )".format(col, t),
               'alias': min_dict,
               'append': True}

APIs['argmax'] = {"argument": ['row', 'header'], 'output': 'row',
                  'function': lambda t, col: nth_maxmin(t, col, order=1, max_or_min="max", arg=True),
                  'tostr': lambda t, col: "argmax {{ {} ; {} }}".format(t, col),
                  'to_nl': lambda t, col: "the row with maximum of ( {} ) of ( {} )".format(col, t),
                  'alias': max_dict,
                  'append': False}

APIs['argmin'] = {"argument": ['row', 'header'], 'output': 'row',
                  'function': lambda t, col: nth_maxmin(t, col, order=1, max_or_min="min", arg=True),
                  'tostr': lambda t, col: "argmin {{ {} ; {} }}".format(t, col),
                  'to_nl': lambda t, col: "the row with minimum of ( {} ) of ( {} )".format(col, t),
                  'alias': min_dict,
                  'append': False}

# add for ordinal
APIs['nth_argmax'] = {"argument": ['row', 'header', 'num'], 'output': 'row',
                      'function': lambda t, col, ind: nth_maxmin(t, col, order=ind, max_or_min="max", arg=True),
                      'tostr': lambda t, col, ind: "nth_argmax {{ {} ; {} ; {} }}".format(t, col, ind),
                      'to_nl': lambda t, col, ind: "the row with the ( {} ) largest of ( {} ) of ( {} )".format(ind, col, t),
                      'alias': max_dict,
                      'append': False}

APIs['nth_argmin'] = {"argument": ['row', 'header', 'num'], 'output': 'row',
                      'function': lambda t, col, ind: nth_maxmin(t, col, order=ind, max_or_min="min", arg=True),
                      'tostr': lambda t, col, ind: "nth_argmin {{ {} ; {} ; {} }}".format(t, col, ind),
                      'to_nl': lambda t, col, ind: "the row with the ( {} ) smallest of ( {} ) of ( {} )".format(ind, col, t),
                      'alias': min_dict,
                      'append': False}

APIs['nth_max'] = {"argument": ['row', 'header', 'num'], 'output': 'num',
                   "function": lambda t, col, ind: nth_maxmin(t, col, order=ind, max_or_min="max", arg=False),
                   "tostr": lambda t, col, ind: "nth_max {{ {} ; {} ; {} }}".format(t, col, ind),
                   "to_nl": lambda t, col, ind: "the ( {} ) maximum value of ( {} ) of ( {} )".format(ind, col, t),
                   'alias': max_dict,
                   'append': True}

APIs['nth_min'] = {"argument": ['row', 'header', 'num'], 'output': 'num',
                   "function": lambda t, col, ind: nth_maxmin(t, col, order=ind, max_or_min="min", arg=False),
                   "tostr": lambda t, col, ind: "nth_min {{ {} ; {} ; {} }}".format(t, col, ind),
                   "to_nl": lambda t, col, ind: "the ( {} ) minimum value of ( {} ) of ( {} )".format(ind, col, t),
                   'alias': min_dict,
                   'append': True}

# With only two argument and the first is not row
APIs['diff'] = {"argument": ['obj', 'obj'], 'output': 'str',
                'function': lambda t1, t2: obj_compare(t1, t2, type="diff"),
                'tostr': lambda t1, t2: "diff {{ {} ; {} }}".format(t1, t2),
                'to_nl': lambda t1, t2: "difference between ( {} ) and ( {} )".format(t1, t2),
                'append': True}

APIs['greater'] = {"argument": ['obj', 'obj'], 'output': 'bool',
                   'function': lambda t1, t2: obj_compare(t1, t2, type="greater"),
                   'tostr': lambda t1, t2: "greater {{ {} ; {} }}".format(t1, t2),
                   'to_nl': lambda t1, t2: "( {} ) is greater than ( {} )".format(t1, t2),
                   'alias': greater_dict,
                   'append': False}

APIs['less'] = {"argument": ['obj', 'obj'], 'output': 'bool',
                'function': lambda t1, t2: obj_compare(t1, t2, type="less"),
                'tostr': lambda t1, t2: "less {{ {} ; {} }}".format(t1, t2),
                'to_nl': lambda t1, t2: "( {} ) is less than ( {} )".format(t1, t2),
                'alias': less_dict,
                'append': True}

APIs['eq'] = {"argument": ['obj', 'obj'], 'output': 'bool',
              'function': lambda t1, t2: obj_compare(t1, t2, type="eq"),
              'tostr': lambda t1, t2: "eq {{ {} ; {} }}".format(t1, t2),
              'to_nl': lambda t1, t2: "( {} ) is equal to ( {} )".format(t1, t2),
              'append': None}

APIs['not_eq'] = {"argument": ['obj', 'obj'], 'output': 'bool',
                  'function': lambda t1, t2: obj_compare(t1, t2, type="not_eq"),
                  'tostr': lambda t1, t2: "not_eq {{ {} ; {} }}".format(t1, t2),
                  'to_nl': lambda t1, t2: "( {} ) is not equal to ( {} )".format(t1, t2),
                  "append": None}

APIs['str_eq'] = {"argument": ['str', 'str'], 'output': 'bool',
                  'function': lambda t1, t2: t1 in t2 or t2 in t1,
                  'tostr': lambda t1, t2: "eq {{ {} ; {} }}".format(t1, t2),
                  'to_nl': lambda t1, t2: "( {} ) is the same as ( {} )".format(t1, t2),
                  "append": None}

APIs['not_str_eq'] = {"argument": ['str', 'str'], 'output': 'bool',
                      'function': lambda t1, t2: t1 not in t2 and t2 not in t1,
                      'tostr': lambda t1, t2: "not_eq {{ {} ; {} }}".format(t1, t2),
                      'to_nl': lambda t1, t2: "( {} ) is not the same as ( {} )".format(t1, t2),
                      "append": None}

APIs['round_eq'] = {"argument": ['obj', 'obj'], 'output': 'bool',
                    'function': lambda t1, t2: obj_compare(t1, t2, round=True, type="eq"),
                    'tostr': lambda t1, t2: "round_eq {{ {} ; {} }}".format(t1, t2),
                    'to_nl': lambda t1, t2: "( {} ) is about ( {} )".format(t1, t2),
                    'append': None}

APIs['and'] = {"argument": ['bool', 'bool'], 'output': 'bool',
               'function': lambda t1, t2: t1 and t2,
               'tostr': lambda t1, t2: "and {{ {} ; {} }}".format(t1, t2),
               'to_nl': lambda t1, t2: "both ( {} ) and ( {} ) are true".format(t1, t2),
               "append": None}

# With only three argument and the first is row
# str
APIs["filter_str_eq"] = {"argument": ['row', 'header', 'str'], "output": "row",
                         "function": lambda t, col, value: fuzzy_match_filter(t, col, value),
                         "tostr": lambda t, col, value: "filter_eq {{ {} ; {} ; {} }}".format(t, col, value),
                         "to_nl": lambda t, col, value: "( {} ) of ( {} ) that fuzzy matches ( {} )".format(col, t, value),
                         'append': False}

APIs["filter_str_not_eq"] = {"argument": ['row', 'header', 'str'], "output": "row",
                             "function": lambda t, col, value: fuzzy_match_filter(t, col, value, negate=True),
                             "tostr": lambda t, col, value: "filter_not_eq {{ {} ; {} ; {} }}".format(t, col, value),
                             "to_nl": lambda t, col, value: "( {} ) of ( {} ) that does not fuzzy matches ( {} )".format(col, t, value),
                             'append': False}

# obj: num or str
APIs["filter_eq"] = {"argument": ['row', 'header', 'obj'], "output": "row",
                     "function": lambda t, col, value: fuzzy_compare_filter(t, col, value, type="eq"),
                     "tostr": lambda t, col, value: "filter_eq {{ {} ; {} ; {} }}".format(t, col, value),
                     "to_nl": lambda t, col, value: "( {} ) of ( {} ) that fuzzy equals ( {} )".format(col, t, value),
                     'append': False}

APIs["filter_not_eq"] = {"argument": ['row', 'header', 'obj'], "output": "row",
                         "function": lambda t, col, value: fuzzy_compare_filter(t, col, value, type="not_eq"),
                         "tostr": lambda t, col, value: "filter_not_eq {{ {} ; {} ; {} }}".format(t, col, value),
                         "to_nl": lambda t, col, value: "( {} ) of ( {} ) that does not fuzzy equals ( {} )".format(col, t, value),
                         'append': False}

APIs["filter_less"] = {"argument": ['row', 'header', 'obj'], "output": "row",
                       "function": lambda t, col, value: fuzzy_compare_filter(t, col, value, type="less"),
                       "tostr": lambda t, col, value: "filter_less {{ {} ; {} ; {} }}".format(t, col, value),
                       "to_nl": lambda t, col, value: "( {} ) of ( {} ) that is fuzzy less than ( {} )".format(col, t, value),
                        'alias': less_dict,
                       "append": False}

APIs["filter_greater"] = {"argument": ['row', 'header', 'obj'], "output": "row",
                          "function": lambda t, col, value: fuzzy_compare_filter(t, col, value, type="greater"),
                          "tostr": lambda t, col, value: "filter_greater {{ {} ; {} ; {} }}".format(t, col, value),
                          "to_nl": lambda t, col, value: "( {} ) of ( {} ) that is fuzzy greater than ( {} )".format(col, t, value),
                          'alias': greater_dict,
                          "append": False}

APIs["filter_greater_eq"] = {"argument": ['row', 'header', 'obj'], "output": "row",
                             "function": lambda t, col, value: fuzzy_compare_filter(t, col, value, type="greater_eq"),
                             "tostr": lambda t, col, value: "filter_greater_eq {{ {} ; {} ; {} }}".format(t, col,value),
                             'alias': greater_dict,
                             "to_nl": lambda t, col, value: "( {} ) of ( {} ) that is fuzzy great than or equal to ( {} )".format(col, t, value),
                             "append": False}

APIs["filter_less_eq"] = {"argument": ['row', 'header', 'obj'], "output": "row",
                          "function": lambda t, col, value: fuzzy_compare_filter(t, col, value, type="less_eq"),
                          "tostr": lambda t, col, value: "filter_less_eq {{ {} ; {} ; {} }}".format(t, col, value),
                          "to_nl": lambda t, col, value: "( {} ) of ( {} ) that is fuzzy less than or equal to ( {} )".format(col, t, value),
                          "alias": less_dict,
                          "append": False}

APIs["filter_all"] = {"argument": ['row', 'header'], "output": "row",
                      "function": lambda t, col: t,
                      "tostr": lambda t, col: "filter_all {{ {} ; {} }}".format(t, col),
                      "to_nl": lambda t, col: "all ( {} ) of ( {} )".format(col, t),
                      'append': False}

# all
# str
APIs["all_str_eq"] = {"argument": ['row', 'header', 'str'], "output": "bool",
                      "function": lambda t, col, value: len(t) == len(fuzzy_match_filter(t, col, value)),
                      "tostr": lambda t, col, value: "all_eq {{ {} ; {} ; {} }}".format(t, col, value),
                      "to_nl": lambda t, col, value: "all ( {} ) of ( {} ) is ( {} )".format(col, t, value),
                      "append": None}

APIs["all_str_not_eq"] = {"argument": ['row', 'header', 'str'], "output": "bool",
                          "function": lambda t, col, value: 0 == len(fuzzy_match_filter(t, col, value)),
                          "to_nl": lambda t, col, value: "all_not_eq {{ {} ; {} ; {} }}".format(t, col, value),
                          "to_nl": lambda t, col, value: "all ( {} ) of ( {} ) is not ( {} )".format(col, t, value),
                          "append": None}

# obj: num or str
APIs["all_eq"] = {"argument": ['row', 'header', 'obj'], "output": "bool",
                  "function": lambda t, col, value: len(t) == len(fuzzy_compare_filter(t, col, value, type="eq")),
                  "tostr": lambda t, col, value: "all_eq {{ {} ; {} ; {} }}".format(t, col, value),
                  "to_nl": lambda t, col, value: "all ( {} ) of ( {} ) is about ( {} )".format(col, t, value),
                  "append": None}

APIs["all_not_eq"] = {"argument": ['row', 'header', 'obj'], "output": "bool",
                      "function": lambda t, col, value: 0 == len(fuzzy_compare_filter(t, col, value, type="eq")),
                      "tostr": lambda t, col, value: "all_not_eq {{ {} ; {} ; {} }}".format(t, col, value),
                      "to_nl": lambda t, col, value: "all ( {} ) of ( {} ) is not about ( {} )".format(col, t, value),
                      "append": None}

APIs["all_less"] = {"argument": ['row', 'header', 'obj'], "output": "bool",
                    "function": lambda t, col, value: len(t) == len(fuzzy_compare_filter(t, col, value, type="less")),
                    "tostr": lambda t, col, value: "all_less {{ {} ; {} ; {} }}".format(t, col, value),
                    "to_nl": lambda t, col, value: "all ( {} ) of ( {} ) is less than ( {} )".format(col, t, value),
                    "alias": less_dict,
                    "append": None}

APIs["all_less_eq"] = {"argument": ['row', 'header', 'obj'], "output": "bool",
                       "function": lambda t, col, value: len(t) == len(
                           fuzzy_compare_filter(t, col, value, type="less_eq")),
                       "tostr": lambda t, col, value: "all_less_eq {{ {} ; {} ; {} }}".format(t, col, value),
                       "to_nl": lambda t, col, value: "all ( {} ) of ( {} ) is less than or equal to ( {} )".format(col, t, value),
                        "alias": less_dict,
                       "append": None}

APIs["all_greater"] = {"argument": ['row', 'header', 'obj'], "output": "bool",
                       "function": lambda t, col, value: len(t) == len(
                           fuzzy_compare_filter(t, col, value, type="greater")),
                       "tostr": lambda t, col, value: "all_greater {{ {} ; {} ; {} }}".format(t, col, value),
                       "to_nl": lambda t, col, value: "all ( {} ) of ( {} ) is greater than ( {} )".format(col, t, value),
                        "alias": greater_dict,
                       "append": None}

APIs["all_greater_eq"] = {"argument": ['row', 'header', 'obj'], "output": "bool",
                          "function": lambda t, col, value: len(t) == len(
                              fuzzy_compare_filter(t, col, value, type="greater_eq")),
                          "tostr": lambda t, col, value: "all_greater_eq {{ {} ; {} ; {} }}".format(t, col, value),
                          "to_nl": lambda t, col, value: "all ( {} ) of ( {} ) is greater or equal to ( {} )".format(col, t, value),
                          "alias": greater_dict,
                          "append": None}

# most
# str
APIs["most_str_eq"] = {"argument": ['row', 'header', 'str'], "output": "bool",
                       "function": lambda t, col, value: len(t) // 3 <= len(fuzzy_match_filter(t, col, value)),
                       "tostr": lambda t, col, value: "most_eq {{ {} ; {} ; {} }}".format(t, col, value),
                       "to_nl": lambda t, col, value: "most of ( {} ) of ( {} ) is ( {} )".format(col, t, value),
                       "alias": most_dict,
                       "append": None}

APIs["most_str_not_eq"] = {"argument": ['row', 'header', 'str'], "output": "bool",
                           "function": lambda t, col, value: len(t) // 3 > len(fuzzy_match_filter(t, col, value)),
                           "tostr": lambda t, col, value: "most_not_eq {{ {} ; {} ; {} }}".format(t, col, value),
                           "to_nl": lambda t, col, value: "most ( {} ) of ( {} ) is not ( {} )".format(col, t, value),
                           "alias": most_dict,
                           "append": None}

# obj: num or str
APIs["most_eq"] = {"argument": ['row', 'header', 'obj'], "output": "bool",
                   "function": lambda t, col, value: len(t) // 3 <= len(fuzzy_compare_filter(t, col, value, type="eq")),
                   "tostr": lambda t, col, value: "most_eq {{ {} ; {} ; {} }}".format(t, col, value),
                   "alias": most_dict,
                   "to_nl": lambda t, col, value: "most of ( {} ) of ( {} ) is ( {} )".format(col, t, value),
                   "append": None}

APIs["most_not_eq"] = {"argument": ['row', 'header', 'obj'], "output": "bool",
                       "function": lambda t, col, value: len(t) // 3 > len(
                           fuzzy_compare_filter(t, col, value, type="eq")),
                       "tostr": lambda t, col, value: "most_not_eq {{ {} ; {} ; {} }}".format(t, col, value),
                       "to_nl": lambda t, col, value: "most of ( {} ) of ( {} ) is not ( {} )".format(col, t, value),
                       "alias": most_dict,
                       "append": None}

APIs["most_less"] = {"argument": ['row', 'header', 'obj'], "output": "bool",
                     "function": lambda t, col, value: len(t) // 3 <= len(
                         fuzzy_compare_filter(t, col, value, type="less")),
                     "tostr": lambda t, col, value: "most_less {{ {} ; {} ; {} }}".format(t, col, value),
                     "to_nl": lambda t, col, value: "most of ( {} ) of ( {} ) is less than ( {} )".format(col, t, value),
                     "alias": less_dict + most_dict,
                     "append": None}

APIs["most_less_eq"] = {"argument": ['row', 'header', 'obj'], "output": "bool",
                        "function": lambda t, col, value: len(t) // 3 <= len(
                            fuzzy_compare_filter(t, col, value, type="less_eq")),
                        "tostr": lambda t, col, value: "most_less_eq {{ {} ; {} ; {} }}".format(t, col, value),
                        "to_nl": lambda t, col, value: "most of ( {} ) of ( {} ) is less than or equal to ( {} )".format(col, t, value),
                        "alias": most_dict + less_dict,
                        "append": None}

APIs["most_greater"] = {"argument": ['row', 'header', 'obj'], "output": "bool",
                        "function": lambda t, col, value: len(t) // 3 <= len(
                            fuzzy_compare_filter(t, col, value, type="greater")),
                        "tostr": lambda t, col, value: "most_greater {{ {} ; {} ; {} }}".format(t, col, value),
                        "to_nl": lambda t, col, value: "most of ( {} ) of ( {} ) is greater than ( {} )".format(col, t, value),
                        "alias": most_dict + greater_dict,
                        "append": None}

APIs["most_greater_eq"] = {"argument": ['row', 'header', 'obj'], "output": "bool",
                           "function": lambda t, col, value: len(t) // 3 <= len(
                               fuzzy_compare_filter(t, col, value, type="greater_eq")),
                           "tostr": lambda t, col, value: "most_greater_eq {{ {} ; {} ; {} }}".format(t, col, value),
                           "to_nl": lambda t, col, value: "most of ( {} ) of ( {} ) is greater than or equal to ( {} )".format(col, t, value),
                           "alias": most_dict + greater_dict,
                           "append": None}

month_map = {'january': 1, 'february': 2, 'march': 3, 'april': 4, 'may': 5, 'june': 6,
             'july': 7, 'august': 8, 'september': 9, 'october': 10, 'november': 11, 'december': 12,
             'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6, 'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10,
             'nov': 11, 'dec': 12}

### regex list

# number format: 
'''
10
1.12
1,000,000
10:00
1st, 2nd, 3rd, 4th
'''
pat_num = r"([-+]?\s?\d*(?:\s?[:,.]\s?\d+)+\b|[-+]?\s?\d+\b|\d+\s?(?=st|nd|rd|th))"

pat_add = r"((?<==\s)\d+)"

# dates
pat_year = r"\b(\d\d\d\d)\b"
pat_day = r"\b(\d\d?)\b"
pat_month = r"\b((?:jan(?:uary)?|feb(?:ruary)?|mar(?:rch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?))\b"


class ExeError(object):
    def __init__(self, message="exe error"):
        self.message = message


### for filter functions. we reset index for the result

# filter_str_eq / not_eq
def fuzzy_match_filter(t, col, val, negate=False):
    trim_t = t[col].str.replace(" ", "")
    trim_val = val.replace(" ", "")

    if negate:
        res = t[~trim_t.str.contains(trim_val, regex=False)]
    else:
        res = t[trim_t.str.contains(trim_val, regex=False)]
    res = res.reset_index(drop=True)
    return res


# filter nums ...
def fuzzy_compare_filter(t, col, val, type):
    '''
    fuzzy compare and filter out rows.
    return empty pd if invalid

    type: eq, not_eq, greater, greater_eq, less, less_eq
    '''

    t[col] = t[col].astype('str')

    # dates
    if len(re.findall(pat_month, val)) > 0:
        year_list = t[col].str.extract(pat_year, expand=False)
        day_list = t[col].str.extract(pat_day, expand=False)
        month_list = t[col].str.extract(pat_month, expand=False)
        month_num_list = month_list.map(month_map)

        # pandas at most 2262
        year_list = year_list.fillna("2260").astype("int")
        day_list = day_list.fillna("1").astype("int")
        month_num_list = month_num_list.fillna("1").astype("int")

        # print (year_list)
        # print (day_list)
        # print (month_num_list)

        date_frame = pd.to_datetime(pd.DataFrame({'year': year_list, 'month': month_num_list, 'day': day_list}))
        # print (date_frame)

        # for val
        year_val = re.findall(pat_year, val)
        if len(year_val) == 0:
            year_val = year_list.iloc[0]
        else:
            year_val = int(year_val[0])

        day_val = re.findall(pat_day, val)
        if len(day_val) == 0:
            day_val = day_list.iloc[0]
        else:
            day_val = int(day_val[0])

        month_val = re.findall(pat_month, val)
        if len(month_val) == 0:
            month_val = month_num_list.iloc[0]
        else:
            month_val = month_map[month_val[0]]

        date_val = pd.datetime(year_val, month_val, day_val)
        # print (date_val)

        if type == "greater":
            res = t[date_frame > date_val]
        elif type == "greater_eq":
            res = t[date_frame >= date_val]
        elif type == "less":
            res = t[date_frame < date_val]
        elif type == "less_eq":
            res = t[date_frame <= date_val]
        elif type == "eq":
            res = t[date_frame == date_val]
        elif type == "not_eq":
            res = t[~date_frame != date_val]

        res = res.reset_index(drop=True)
        return res

    # numbers, or mixed numbers and strings
    val_pat = re.findall(pat_num, val)
    if len(val_pat) == 0:
        # return pd.DataFrame(columns=list(t.columns))
        # fall back to full string matching
        if type == "eq":
            return t[t[col].str.contains(val, regex=False)]
        elif type == "not_eq":
            return t[~t[col].str.contains(val, regex=False)]
        else:
            return pd.DataFrame(columns=list(t.columns))

        # return pd.DataFrame(columns=list(t.columns))

    num = val_pat[0].replace(",", "")
    num = num.replace(":", "")
    num = num.replace(" ", "")
    try:
        num = float(num)
    except:
        num = num.replace(".", "")
        num = float(num)
    # print (num)

    pats = t[col].str.extract(pat_add, expand=False)
    if pats.isnull().all():
        pats = t[col].str.extract(pat_num, expand=False)
    if pats.isnull().all():
        return pd.DataFrame(columns=list(t.columns))
    nums = pats.str.replace(",", "")
    nums = nums.str.replace(":", "")
    nums = nums.str.replace(" ", "")
    try:
        nums = nums.astype("float")
    except:
        nums = nums.str.replace(".", "")
        nums = nums.astype("float")
    # print (nums)

    if type == "greater":
        res = t[np.greater(nums, num)]
    elif type == "greater_eq":
        res = t[np.greater_equal(nums, num)]
    elif type == "less":
        res = t[np.less(nums, num)]
    elif type == "less_eq":
        res = t[np.less_equal(nums, num)]
    elif type == "eq":
        res = t[np.isclose(nums, num)]
    elif type == "not_eq":
        res = t[~np.isclose(nums, num)]

    res = res.reset_index(drop=True)
    return res

    # all invalid
    return pd.DataFrame(columns=list(t.columns))


### for comparison
def obj_compare(num1, num2, round=False, type="eq"):
    tolerance = 0.15 if round else 1e-9
    # both numeric
    try:
        num_1 = float(num1)
        num_2 = float(num2)

        # if negate:
        #   return (not math.isclose(num_1, num_2, rel_tol=tolerance))
        # return math.isclose(num_1, num_2, rel_tol=tolerance)

        if type == "eq":
            return math.isclose(num_1, num_2, rel_tol=tolerance)
        elif type == "not_eq":
            return (not math.isclose(num_1, num_2, rel_tol=tolerance))
        elif type == "greater":
            return num_1 > num_2
        elif type == "less":
            return num_1 < num_2
        elif type == "diff":
            return num_1 - num_2




    except ValueError:
        # strings
        # mixed numbers and strings
        num1 = str(num1)
        num2 = str(num2)

        # dates
        # num1
        if len(re.findall(pat_month, num1)) > 0:
            year_val1 = re.findall(pat_year, num1)
            if len(year_val1) == 0:
                year_val1 = int("2260")
            else:
                year_val1 = int(year_val1[0])

            day_val1 = re.findall(pat_day, num1)
            if len(day_val1) == 0:
                day_val1 = int("1")
            else:
                day_val1 = int(day_val1[0])

            month_val1 = re.findall(pat_month, num1)
            if len(month_val1) == 0:
                month_val1 = int("1")
            else:
                month_val1 = month_map[month_val1[0]]

            try:
                date_val1 = pd.datetime(year_val1, month_val1, day_val1)
            except:
                return ExeError

            # num2
            year_val2 = re.findall(pat_year, num2)
            if len(year_val2) == 0:
                year_val2 = int("2260")
            else:
                year_val2 = int(year_val2[0])

            day_val2 = re.findall(pat_day, num2)
            if len(day_val2) == 0:
                day_val2 = int("1")
            else:
                day_val2 = int(day_val2[0])

            month_val2 = re.findall(pat_month, num2)
            if len(month_val2) == 0:
                month_val2 = int("1")
            else:
                month_val2 = month_map[month_val2[0]]

            try:
                date_val2 = pd.datetime(year_val2, month_val2, day_val2)
            except:
                return ExeError

            # if negate:
            #   return date_val1 != date_val2
            # else:
            #   return date_val1 == date_val2

            if type == "eq":
                return date_val1 == date_val2
            elif type == "not_eq":
                return date_val1 != date_val2
            elif type == "greater":
                return date_val1 > date_val2
            elif type == "less":
                return date_val1 < date_val2
            # for diff return string
            elif type == "diff":
                return str((date_val1 - date_val2).days) + " days"

        # mixed string and numerical
        val_pat1 = re.findall(pat_num, num1)
        val_pat2 = re.findall(pat_num, num2)
        if len(val_pat1) == 0 or len(val_pat2) == 0:

            # fall back to full string matching
            if type == "not_eq":
                return (num1 not in num2) and (num2 not in num1)
            elif type == "eq":
                return num1 in num2 or num2 in num1
            else:
                return ExeError()

        num_1 = val_pat1[0].replace(",", "")
        num_1 = num_1.replace(":", "")
        num_1 = num_1.replace(" ", "")
        try:
            num_1 = float(num_1)
        except:
            num_1 = num_1.replace(".", "")
            num_1 = float(num_1)

        num_2 = val_pat2[0].replace(",", "")
        num_2 = num_2.replace(":", "")
        num_2 = num_2.replace(" ", "")
        try:
            num_2 = float(num_2)
        except:
            num_2 = num_2.replace(".", "")
            num_2 = float(num_2)

        # if negate:
        #   return (not math.isclose(num_1, num_2, rel_tol=tolerance))
        # return math.isclose(num_1, num_2, rel_tol=tolerance)

        if type == "eq":
            return math.isclose(num_1, num_2, rel_tol=tolerance)
        elif type == "not_eq":
            return (not math.isclose(num_1, num_2, rel_tol=tolerance))
        elif type == "greater":
            return num_1 > num_2
        elif type == "less":
            return num_1 < num_2
        elif type == "diff":
            return num_1 - num_2


### for aggregation: sum avg

def agg(t, col, type):
    '''
    sum or avg for aggregation
    '''

    # unused
    if t.dtypes[col] == np.int64 or t.dtypes[col] == np.float64:
        if type == "sum":
            res = t[col].sum()
        elif type == "avg":
            res = t[col].mean()

        return res

    else:

        pats = t[col].str.extract(pat_add, expand=False)
        if pats.isnull().all():
            pats = t[col].str.extract(pat_num, expand=False)
        if pats.isnull().all():
            return 0.0
        pats.fillna("0.0")
        nums = pats.str.replace(",", "")
        nums = nums.str.replace(":", "")
        nums = nums.str.replace(" ", "")
        try:
            nums = nums.astype("float")
        except:
            nums = nums.str.replace(".", "")
            nums = nums.astype("float")

        # print (nums)
        if type == "sum":
            return nums.sum()
        elif type == "mean":
            return nums.mean()


### for hop

def hop_op(t, col):
    if len(t) == 0:
        return ExeError()

    return t[col].values[0]


### for superlative, ordinal

def nth_maxmin(t, col, order=1, max_or_min="max", arg=False):
    '''
    for max, min, argmax, argmin,
    nth_max, nth_min, nth_argmax, nth_argmin

    return string or rows
    '''

    order = int(order)
    ### return the original content for max,min
    # dates
    date_pats = t[col].str.extract(pat_month, expand=False)
    if not date_pats.isnull().all():
        year_list = t[col].str.extract(pat_year, expand=False)
        day_list = t[col].str.extract(pat_day, expand=False)
        month_list = t[col].str.extract(pat_month, expand=False)
        month_num_list = month_list.map(month_map)

        # pandas at most 2262
        year_list = year_list.fillna("2260").astype("int")
        day_list = day_list.fillna("1").astype("int")
        month_num_list = month_num_list.fillna("1").astype("int")

        # print (year_list)
        # print (day_list)
        # print (month_num_list)

        try:
            date_series = pd.to_datetime(pd.DataFrame({'year': year_list, 'month': month_num_list, 'day': day_list}))
            # print (date_series)

            if max_or_min == "max":
                tar_row = date_series.nlargest(order).iloc[[-1]]
            elif max_or_min == "min":
                tar_row = date_series.nsmallest(order).iloc[[-1]]
            ind = list(tar_row.index.values)
            if arg:
                res = t.iloc[ind]
            else:
                res = t.iloc[ind][col].values[0]

            return res

        except:
            pass

    # mixed string and numerical
    pats = t[col].str.extract(pat_add, expand=False)
    if pats.isnull().all():
        pats = t[col].str.extract(pat_num, expand=False)
    if pats.isnull().all():
        return ExeError()
    nums = pats.str.replace(",", "")
    nums = nums.str.replace(":", "")
    nums = nums.str.replace(" ", "")
    try:
        nums = nums.astype("float")
    except:
        nums = nums.str.replace(".", "")
        nums = nums.astype("float")

    try:
        if max_or_min == "max":
            tar_row = nums.nlargest(order).iloc[[-1]]
        elif max_or_min == "min":
            tar_row = nums.nsmallest(order).iloc[[-1]]
        ind = list(tar_row.index.values)
        # print (ind)
        # print (t.iloc[ind][col].values)
        if arg:
            res = t.iloc[ind]
        else:
            res = t.iloc[ind][col].values[0]

    except:
        return ExeError()

    # print (res)
    return res


def is_ascii(s):
    return all(ord(c) < 128 for c in s)

from typing import List
import re
import sqlparse


class TreeNode(object):
    def __init__(self, name=None, father=None):
        self.name: str = name
        self.rename: str = name
        self.father: TreeNode = father
        self.children: List = []
        self.produced_col_name_s = None

    def __eq__(self, other):
        return self.rename == other.rename

    def __hash__(self):
        return hash(self.rename)

    def set_name(self, name):
        self.name = name
        self.rename = name

    def add_child(self, child):
        self.children.append(child)
        child.father = self

    def rename_father_col(self, col_idx: int, col_prefix: str = "col_"):
        new_col_name = "{}{}".format(col_prefix, col_idx)
        self.father.rename = self.father.rename.replace(self.name, "{}".format(new_col_name))
        self.produced_col_name_s = [new_col_name]  # fixme when multiple outputs for a qa func

    def rename_father_val(self, val_names):
        if len(val_names) == 1:
            val_name = val_names[0]
            new_val_equals_str = "'{}'".format(val_name) if isinstance(convert_type(val_name), str) else "{}".format(
                val_name)
        else:
            new_val_equals_str = '({})'.format(', '.join(["'{}'".format(val_name) for val_name in val_names]))
        self.father.rename = self.father.rename.replace(self.name, new_val_equals_str)


def get_cfg_tree(nsql: str):
    """
    Parse QA() into a tree for execution guiding.
    @param nsql:
    @return:
    """

    stack: List = []  # Saving the state of the char.
    expression_stack: List = []  # Saving the state of the expression.
    current_tree_node = TreeNode(name=nsql)

    for idx in range(len(nsql)):
        if nsql[idx] == "(":
            stack.append(idx)
            if idx > 1 and nsql[idx - 2:idx + 1] == "QA(" and idx - 2 != 0:
                tree_node = TreeNode()
                current_tree_node.add_child(tree_node)
                expression_stack.append(current_tree_node)
                current_tree_node = tree_node
        elif nsql[idx] == ")":
            left_clause_idx = stack.pop()
            if idx > 1 and nsql[left_clause_idx - 2:left_clause_idx + 1] == "QA(" and left_clause_idx - 2 != 0:
                # the QA clause
                nsql_span = nsql[left_clause_idx - 2:idx + 1]
                current_tree_node.set_name(nsql_span)
                current_tree_node = expression_stack.pop()

    return current_tree_node


def get_steps(tree_node: TreeNode, steps: List):
    """Pred-Order Traversal"""
    for child in tree_node.children:
        get_steps(child, steps)
    steps.append(tree_node)


def parse_question_paras(nsql: str, qa_model):
    # We assume there's no nested qa inside when running this func
    nsql = nsql.strip(" ;")
    assert nsql[:3] == "QA(" and nsql[-1] == ")", "must start with QA( symbol and end with )"
    assert not "QA" in nsql[2:-1],  "must have no nested qa inside"

    # Get question and the left part(paras_raw_str)
    all_quote_idx = [i.start() for i in re.finditer('\"', nsql)]
    question = nsql[all_quote_idx[0] + 1: all_quote_idx[1]]
    paras_raw_str = nsql[all_quote_idx[1] + 1:-1].strip(" ;")

    # Split Parameters(SQL/column/value) from all parameters.
    paras = [_para.strip(' ;') for _para in sqlparse.split(paras_raw_str)]
    return question, paras


def convert_type(value):
    try:
        return eval(value)
    except Exception as e:
        return value


def nsql_role_recognize(nsql_like_str, all_headers, all_passage_titles, all_image_titles):
    """Recognize role. (SQL/column/value) """
    orig_nsql_like_str = nsql_like_str

    # strip the first and the last '`'
    if nsql_like_str.startswith('`') and nsql_like_str.endswith('`'):
        nsql_like_str = nsql_like_str[1:-1]

    # Case 1: if col in header, it is column type.
    if nsql_like_str in all_headers or nsql_like_str in list(map(lambda x: x.lower(), all_headers)):
        return 'col', orig_nsql_like_str

    # fixme: add case when the this nsql_like_str both in table headers, images title and in passages title.
    # Case 2.1: if it is title of certain passage.
    if (nsql_like_str.lower() in list(map(lambda x: x.lower(), all_passage_titles))) \
            and (nsql_like_str.lower() in list(map(lambda x: x.lower(), all_image_titles))):
        return "passage_title_and_image_title", orig_nsql_like_str
    else:
        try:
            nsql_like_str_evaled = str(eval(nsql_like_str))
            if (nsql_like_str_evaled.lower() in list(map(lambda x: x.lower(), all_passage_titles))) \
                    and (nsql_like_str_evaled.lower() in list(map(lambda x: x.lower(), all_image_titles))):
                return "passage_title_and_image_title", nsql_like_str_evaled
        except:
            pass

    # Case 2.2: if it is title of certain passage.
    if nsql_like_str.lower() in list(map(lambda x: x.lower(), all_passage_titles)):
        return "passage_title", orig_nsql_like_str
    else:
        try:
            nsql_like_str_evaled = str(eval(nsql_like_str))
            if nsql_like_str_evaled.lower() in list(map(lambda x: x.lower(), all_passage_titles)):
                return "passage_title", nsql_like_str_evaled
        except:
            pass

    # Case 2.3: if it is title of certain picture.
    if nsql_like_str.lower() in list(map(lambda x: x.lower(), all_image_titles)):
        return "image_title", orig_nsql_like_str
    else:
        try:
            nsql_like_str_evaled = str(eval(nsql_like_str))
            if nsql_like_str_evaled.lower() in list(map(lambda x: x.lower(), all_image_titles)):
                return "image_title", nsql_like_str_evaled
        except:
            pass

    # Case 4: if it can be parsed by eval(), it is value type.
    try:
        eval(nsql_like_str)
        return 'val', orig_nsql_like_str
    except Exception as e:
        pass

    # Case 5: else it should be the sql, if it isn't, exception will be raised.
    return 'complete_sql', orig_nsql_like_str


def remove_duplicate(original_list):
    no_duplicate_list = []
    [no_duplicate_list.append(i) for i in original_list if i not in no_duplicate_list]
    return no_duplicate_list


def extract_answers(sub_table):
    if not sub_table or sub_table['header'] is None:
        return []
    answer = []
    if 'row_id' in sub_table['header']:
        for _row in sub_table['rows']:
            answer.extend(_row[1:])
        return answer
    else:
        for _row in sub_table['rows']:
            answer.extend(_row)
        return answer

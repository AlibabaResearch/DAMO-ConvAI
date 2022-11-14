from .APIs import *
from collections import defaultdict
import random


class Node(object):
    # static member
    swap_dict = defaultdict(list)
    for op, attr in APIs.items():
        swap_dict[' '.join(attr['argument'])].append(op)

    def __init__(self, full_table, dict_in):
        '''
        construct tree
        '''

        self.full_table = full_table
        self.dict_in = dict_in
        self.func = dict_in["func"]

        # row, num, str, obj, header, bool
        self.arg_type_list = APIs[self.func]["argument"]
        self.arg_list = []

        # [("text_node", a), ("func_node", b)]
        self.child_list = []
        child_list = dict_in["args"]

        if len(self.arg_type_list) != len(child_list):
            assert len(self.arg_type_list) != len(child_list)

        # bool, num, str, row
        self.out_type = APIs[self.func]["output"]

        for each_child in child_list:
            if isinstance(each_child, str):
                self.child_list.append(("text_node", each_child))
            elif isinstance(each_child, dict):
                sub_func_node = Node(self.full_table, each_child)
                self.child_list.append(("func_node", sub_func_node))
            else:
                raise ValueError("child type error")

        self.result = None

    def eval(self):
        arg_list = []
        for each_child, each_type in zip(self.child_list, self.arg_type_list):
            if each_child[0] == "text_node":
                if each_child[1] == "all_rows":
                    arg_list.append(self.full_table)
                else:
                    arg_list.append(each_child[1])
            else:
                sub_result = each_child[1].eval()
                # print ("exit func: ", each_child[1].func)

                # invalid
                if isinstance(sub_result, ExeError):
                    return "ExeError(): sublevel error"
                elif each_type == "row":
                    if not isinstance(sub_result, pd.DataFrame):
                        return "ExeError(): error function return type"
                elif each_type == "bool":
                    if not isinstance(sub_result, bool):
                        return "ExeError(): error function return type"
                elif each_type == "str":
                    if not isinstance(sub_result, str):
                        return "ExeError(): error function return type"

                arg_list.append(sub_result)

        result = APIs[self.func]["function"](*arg_list)
        return result

    def to_nl(self):
        arg_list = []
        for each_child, each_type in zip(self.child_list, self.arg_type_list):
            if each_child[0] == "text_node":
                if each_child[1] == "all_rows":
                    arg_list.append('all rows')
                else:
                    arg_list.append(each_child[1])
            else:
                sub_result = each_child[1].to_nl()
                # print ("exit func: ", each_child[1].func)

                arg_list.append(sub_result)

        result = APIs[self.func]["to_nl"](*arg_list)
        return result

    def to_code(self):
        arg_list = []
        for each_child, each_type in zip(self.child_list, self.arg_type_list):
            if each_child[0] == "text_node":
                if each_child[1] == "all_rows":
                    arg_list.append('all_rows')
                else:
                    arg_list.append(each_child[1])
            else:
                sub_result = each_child[1].to_code()
                # print ("exit func: ", each_child[1].func)

                arg_list.append(sub_result)

        result = APIs[self.func]["tostr"](*arg_list)
        return result

    def _mutate_dict(self, dict_in, alpha=0.5, beta=0.5, gamma=0.6, theta=0.15, omega=0.2):
        new_dict = {}
        # mutate function
        new_func = dict_in['func']
        if random.random() > alpha:
            for arg, ops in self.swap_dict.items():
                if dict_in['func'] in ops:
                    swap_func = random.choice(ops)  # have chance not changing
                    new_func = swap_func
                    break
        new_dict['func'] = new_func

        # deal with args
        new_dict['args'] = []
        for each_child in dict_in["args"]:
            if isinstance(each_child, str):
                new_child = each_child
                # mutate int
                if each_child.isnumeric() and random.random() < theta:
                    new_child = max(int(each_child) + random.randint(-10, 10), 0)
                    new_child = str(new_child)  # TODO: float numbers

                # mutate columns
                cols = self.full_table.columns
                if each_child in cols:
                    if random.random() > beta:
                        new_child = random.choice(cols)  # have chance not changing
                        # TODO: content mutation
                new_dict['args'].append(new_child)

            elif isinstance(each_child, dict):
                new_child = self._mutate_dict(each_child)
                new_dict['args'].append(new_child)
            else:
                raise ValueError("child type error")

        return new_dict

    def mutate(self, mutate_num_max=500, alpha=0.5, beta=0.5, gamma=0.6, theta=0.15, omega=0.2):
        mutations = []
        visited_node = set()
        for i in range(mutate_num_max):
            new_dict = self._mutate_dict(self.dict_in, alpha=0.5, beta=0.5, gamma=0.6, theta=0.15, omega=0.2)
            if str(new_dict) not in visited_node:
                visited_node.add(str(new_dict))
                new_node = Node(self.full_table, new_dict)
                # test node
                try:
                    new_result = str(new_node.eval())
                except:
                    continue
                if 'ExeError():' not in new_result:
                    mutations.append(new_node)
        return mutations

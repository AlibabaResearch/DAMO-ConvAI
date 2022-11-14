# coding=utf-8

from asdl.asdl import *
from asdl.asdl_ast import AbstractSyntaxTree
from asdl.transition_system import *

class Hypothesis(object):
    def __init__(self):
        self.tree = None
        self.actions = []
        self.score = 0.
        self.frontier_node = None
        self.frontier_field = None
        self._value_buffer = []

        # record the current time step
        self.t = 0

    def apply_action(self, action):
        if self.tree is None: # the first action
            assert isinstance(action, ApplyRuleAction), 'Invalid action [%s], only ApplyRule action is valid ' \
                                                        'at the beginning of decoding'

            self.tree = AbstractSyntaxTree(action.production)
            self.update_frontier_info()
        elif self.frontier_node:
            if isinstance(self.frontier_field.type, ASDLCompositeType):
                if isinstance(action, ApplyRuleAction):
                    field_value = AbstractSyntaxTree(action.production)
                    field_value.created_time = self.t
                    self.frontier_field.add_value(field_value)
                    self.update_frontier_info()
                elif isinstance(action, ReduceAction):
                    assert self.frontier_field.cardinality in ('optional', 'multiple'), 'Reduce action can only be ' \
                                                                                        'applied on field with multiple ' \
                                                                                        'cardinality'
                    self.frontier_field.set_finish()
                    self.update_frontier_info()
                else:
                    raise ValueError('Invalid action [%s] on field [%s]' % (action, self.frontier_field))
            else:  # fill in a primitive field
                if isinstance(action, GenTokenAction):
                    # only field of type string requires termination signal </primitive>
                    end_primitive = False
                    if self.frontier_field.type.name == 'string':
                        if action.is_stop_signal():
                            self.frontier_field.add_value(' '.join(self._value_buffer))
                            self._value_buffer = []

                            end_primitive = True
                        else:
                            self._value_buffer.append(action.token)
                    else:
                        self.frontier_field.add_value(action.token)
                        end_primitive = True

                    if end_primitive and self.frontier_field.cardinality in ('single', 'optional'):
                        self.frontier_field.set_finish()
                        self.update_frontier_info()

                elif isinstance(action, ReduceAction):
                    assert self.frontier_field.cardinality in ('optional', 'multiple'), 'Reduce action can only be ' \
                                                                                        'applied on field with multiple ' \
                                                                                        'cardinality'
                    self.frontier_field.set_finish()
                    self.update_frontier_info()
                else:
                    raise ValueError('Can only invoke GenToken or Reduce actions on primitive fields')

        self.t += 1
        self.actions.append(action)

    def update_frontier_info(self):
        def _find_frontier_node_and_field(tree_node):
            # return None if each field of this ast node is realized else unfinished ast node, unrealized field
            if tree_node:
                for field in tree_node.fields:
                    # if it's an intermediate node, check its children
                    if isinstance(field.type, ASDLCompositeType) and field.value:
                        if field.cardinality in ('single', 'optional'): iter_values = [field.value]
                        else: iter_values = field.value

                        for child_node in iter_values:
                            result = _find_frontier_node_and_field(child_node)
                            if result: return result

                    # now all its possible children are checked
                    if not field.finished:
                        return tree_node, field

                return None
            else: return None

        frontier_info = _find_frontier_node_and_field(self.tree)
        if frontier_info:
            self.frontier_node, self.frontier_field = frontier_info
        else:
            self.frontier_node, self.frontier_field = None, None

    def clone_and_apply_action(self, action):
        new_hyp = self.copy()
        new_hyp.apply_action(action)

        return new_hyp

    def copy(self):
        new_hyp = Hypothesis()
        if self.tree:
            new_hyp.tree = self.tree.copy()

        new_hyp.actions = list(self.actions)
        new_hyp.score = self.score
        new_hyp._value_buffer = list(self._value_buffer)
        new_hyp.t = self.t

        new_hyp.update_frontier_info()

        return new_hyp

    @property
    def completed(self):
        return self.tree and self.frontier_field is None

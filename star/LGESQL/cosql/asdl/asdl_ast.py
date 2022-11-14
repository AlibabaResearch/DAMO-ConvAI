# coding=utf-8
from io import StringIO
from asdl.asdl import *

class AbstractSyntaxTree(object):
    def __init__(self, production, realized_fields=None):
        self.production = production

        # a child is essentially a *realized_field*
        self.fields = []

        # record its parent field to which it's attached
        self.parent_field = None

        # used in decoding, record the time step when this node was created
        self.created_time = 0

        if realized_fields:
            assert len(realized_fields) == len(self.production.fields)

            for field in realized_fields:
                self.add_child(field)
        else:
            for field in self.production.fields:
                self.add_child(RealizedField(field))

    def add_child(self, realized_field):
        # if isinstance(realized_field.value, AbstractSyntaxTree):
        #     realized_field.value.parent = self
        self.fields.append(realized_field)
        realized_field.parent_node = self

    def __getitem__(self, field_name):
        for field in self.fields:
            if field.name == field_name: return field
        raise KeyError

    def sanity_check(self):
        if len(self.production.fields) != len(self.fields):
            raise ValueError('filed number must match')
        for field, realized_field in zip(self.production.fields, self.fields):
            assert field == realized_field.field
        for child in self.fields:
            for child_val in child.as_value_list:
                if isinstance(child_val, AbstractSyntaxTree):
                    child_val.sanity_check()

    def copy(self):
        new_tree = AbstractSyntaxTree(self.production)
        new_tree.created_time = self.created_time
        for i, old_field in enumerate(self.fields):
            new_field = new_tree.fields[i]
            new_field._not_single_cardinality_finished = old_field._not_single_cardinality_finished
            if isinstance(old_field.type, ASDLCompositeType):
                for value in old_field.as_value_list:
                    new_field.add_value(value.copy())
            else:
                for value in old_field.as_value_list:
                    new_field.add_value(value)

        return new_tree

    def to_string(self, sb=None):
        is_root = False
        if sb is None:
            is_root = True
            sb = StringIO()

        sb.write('(')
        sb.write(self.production.constructor.name)

        for field in self.fields:
            sb.write(' ')
            sb.write('(')
            sb.write(field.type.name)
            sb.write(Field.get_cardinality_repr(field.cardinality))
            sb.write('-')
            sb.write(field.name)

            if field.value is not None:
                for val_node in field.as_value_list:
                    sb.write(' ')
                    if isinstance(field.type, ASDLCompositeType):
                        val_node.to_string(sb)
                    else:
                        sb.write(str(val_node).replace(' ', '-SPACE-'))

            sb.write(')')  # of field

        sb.write(')')  # of node

        if is_root:
            return sb.getvalue()

    def __hash__(self):
        code = hash(self.production)
        for field in self.fields:
            code = code + 37 * hash(field)

        return code

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False

        if self.created_time != other.created_time:
            return False

        if self.production != other.production:
            return False

        if len(self.fields) != len(other.fields):
            return False

        for i in range(len(self.fields)):
            if self.fields[i] != other.fields[i]: return False

        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return repr(self.production)

    @property
    def size(self):
        node_num = 1
        for field in self.fields:
            for val in field.as_value_list:
                if isinstance(val, AbstractSyntaxTree):
                    node_num += val.size
                else: node_num += 1

        return node_num


class RealizedField(Field):
    """wrapper of field realized with values"""
    def __init__(self, field, value=None, parent=None):
        super(RealizedField, self).__init__(field.name, field.type, field.cardinality)

        # record its parent AST node
        self.parent_node = None

        # FIXME: hack, return the field as a property
        self.field = field

        # initialize value to correct type
        if self.cardinality == 'multiple':
            self.value = []
            if value is not None:
                for child_node in value:
                    self.add_value(child_node)
        else:
            self.value = None
            # note the value could be 0!
            if value is not None: self.add_value(value)

        # properties only used in decoding, record if the field is finished generating
        # when card in [optional, multiple]
        self._not_single_cardinality_finished = False

    def add_value(self, value):
        if isinstance(value, AbstractSyntaxTree):
            value.parent_field = self

        if self.cardinality == 'multiple':
            self.value.append(value)
        else:
            self.value = value

    @property
    def as_value_list(self):
        """get value as an iterable"""
        if self.cardinality == 'multiple': return self.value
        elif self.value is not None: return [self.value]
        else: return []

    @property
    def finished(self):
        if self.cardinality == 'single':
            if self.value is None: return False
            else: return True
        elif self.cardinality == 'optional' and self.value is not None:
            return True
        else:
            if self._not_single_cardinality_finished: return True
            else: return False

    def set_finish(self):
        # assert self.cardinality in ('optional', 'multiple')
        self._not_single_cardinality_finished = True

    def __eq__(self, other):
        if super(RealizedField, self).__eq__(other):
            if type(other) == Field: return True  # FIXME: hack, Field and RealizedField can compare!
            if self.value == other.value: return True
            else: return False
        else: return False

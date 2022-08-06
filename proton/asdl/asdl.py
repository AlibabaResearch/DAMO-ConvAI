# coding=utf-8
from collections import OrderedDict, Counter
from itertools import chain
import re, os

def remove_comment(text):
    text = re.sub(re.compile("#.*"), "", text)
    text = '\n'.join(filter(lambda x: x, text.split('\n')))
    return text

class ASDLGrammar(object):
    """
    Collection of types, constructors and productions
    """
    def __init__(self, productions, file_path):
        # productions are indexed by their head types
        file_name = os.path.basename(file_path)
        grammar_name = file_name[:file_name.index('.txt')] if '.txt' in file_name else file_name
        self._grammar_name = grammar_name
        self._productions = OrderedDict()
        self._constructor_production_map = dict()
        for prod in productions:
            if prod.type not in self._productions:
                self._productions[prod.type] = list()
            self._productions[prod.type].append(prod)
            self._constructor_production_map[prod.constructor.name] = prod

        self.root_type = productions[0].type
        # number of constructors
        self.size = sum(len(head) for head in self._productions.values())

        # get entities to their ids map
        self.prod2id = {prod: i for i, prod in enumerate(self.productions)}
        self.type2id = {type: i for i, type in enumerate(self.types)}
        self.field2id = {field: i for i, field in enumerate(self.fields)}

        self.id2prod = {i: prod for i, prod in enumerate(self.productions)}
        self.id2type = {i: type for i, type in enumerate(self.types)}
        self.id2field = {i: field for i, field in enumerate(self.fields)}

    def __len__(self):
        return self.size

    @property
    def productions(self):
        return sorted(chain.from_iterable(self._productions.values()), key=lambda x: repr(x))

    def __getitem__(self, datum):
        if isinstance(datum, str):
            return self._productions[ASDLType(datum)]
        elif isinstance(datum, ASDLType):
            return self._productions[datum]

    def get_prod_by_ctr_name(self, name):
        return self._constructor_production_map[name]

    @property
    def types(self):
        if not hasattr(self, '_types'):
            all_types = set()
            for prod in self.productions:
                all_types.add(prod.type)
                all_types.update(map(lambda x: x.type, prod.constructor.fields))

            self._types = sorted(all_types, key=lambda x: x.name)

        return self._types

    @property
    def fields(self):
        if not hasattr(self, '_fields'):
            all_fields = set()
            for prod in self.productions:
                all_fields.update(prod.constructor.fields)

            self._fields = sorted(all_fields, key=lambda x: (x.name, x.type.name, x.cardinality))

        return self._fields

    @property
    def primitive_types(self):
        return filter(lambda x: isinstance(x, ASDLPrimitiveType), self.types)

    @property
    def composite_types(self):
        return filter(lambda x: isinstance(x, ASDLCompositeType), self.types)

    def is_composite_type(self, asdl_type):
        return asdl_type in self.composite_types

    def is_primitive_type(self, asdl_type):
        return asdl_type in self.primitive_types

    @staticmethod
    def from_filepath(file_path):
        def _parse_field_from_text(_text):
            d = _text.strip().split(' ')
            name = d[1].strip()
            type_str = d[0].strip()
            cardinality = 'single'
            if type_str[-1] == '*':
                type_str = type_str[:-1]
                cardinality = 'multiple'
            elif type_str[-1] == '?':
                type_str = type_str[:-1]
                cardinality = 'optional'

            if type_str in primitive_type_names:
                return Field(name, ASDLPrimitiveType(type_str), cardinality=cardinality)
            else:
                return Field(name, ASDLCompositeType(type_str), cardinality=cardinality)

        def _parse_constructor_from_text(_text):
            _text = _text.strip()
            fields = None
            if '(' in _text:
                name = _text[:_text.find('(')]
                field_blocks = _text[_text.find('(') + 1:_text.find(')')].split(',')
                fields = map(_parse_field_from_text, field_blocks)
            else:
                name = _text

            if name == '': name = None

            return ASDLConstructor(name, fields)

        with open(file_path, 'r') as inf:
            text = inf.read()
        lines = remove_comment(text).split('\n')
        lines = list(map(lambda l: l.strip(), lines))
        lines = list(filter(lambda l: l, lines))
        line_no = 0

        # first line is always the primitive types
        primitive_type_names = list(map(lambda x: x.strip(), lines[line_no].split(',')))
        line_no += 1

        all_productions = list()

        while True:
            type_block = lines[line_no]
            type_name = type_block[:type_block.find('=')].strip()
            constructors_blocks = type_block[type_block.find('=') + 1:].split('|')
            i = line_no + 1
            while i < len(lines) and lines[i].strip().startswith('|'):
                t = lines[i].strip()
                cont_constructors_blocks = t[1:].split('|')
                constructors_blocks.extend(cont_constructors_blocks)

                i += 1

            constructors_blocks = filter(lambda x: x and x.strip(), constructors_blocks)

            # parse type name
            new_type = ASDLPrimitiveType(type_name) if type_name in primitive_type_names else ASDLCompositeType(type_name)
            constructors = map(_parse_constructor_from_text, constructors_blocks)

            productions = list(map(lambda c: ASDLProduction(new_type, c), constructors))
            all_productions.extend(productions)

            line_no = i
            if line_no == len(lines):
                break

        grammar = ASDLGrammar(all_productions, file_path)

        return grammar


class ASDLProduction(object):
    def __init__(self, type, constructor):
        self.type = type
        self.constructor = constructor

    @property
    def fields(self):
        return self.constructor.fields

    def __getitem__(self, field_name):
        return self.constructor[field_name]

    def __hash__(self):
        h = hash(self.type) ^ hash(self.constructor)

        return h

    def __eq__(self, other):
        return isinstance(other, ASDLProduction) and \
               self.type == other.type and \
               self.constructor == other.constructor

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return '%s -> %s' % (self.type.__repr__(plain=True), self.constructor.__repr__(plain=True))


class ASDLConstructor(object):
    def __init__(self, name, fields=None):
        self.name = name
        self.fields = []
        if fields:
            self.fields = list(fields)

    def __getitem__(self, field_name):
        for field in self.fields:
            if field.name == field_name: return field

        raise KeyError

    def __hash__(self):
        h = hash(self.name)
        for field in self.fields:
            h ^= hash(field)

        return h

    def __eq__(self, other):
        return isinstance(other, ASDLConstructor) and \
               self.name == other.name and \
               self.fields == other.fields

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self, plain=False):
        plain_repr = '%s(%s)' % (self.name,
                                 ', '.join(f.__repr__(plain=True) for f in self.fields))
        if plain: return plain_repr
        else: return 'Constructor(%s)' % plain_repr


class Field(object):
    def __init__(self, name, type, cardinality):
        self.name = name
        self.type = type

        assert cardinality in ['single', 'optional', 'multiple']
        self.cardinality = cardinality

    def __hash__(self):
        h = hash(self.name) ^ hash(self.type)
        h ^= hash(self.cardinality)

        return h

    def __eq__(self, other):
        return isinstance(other, Field) and \
               self.name == other.name and \
               self.type == other.type and \
               self.cardinality == other.cardinality

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self, plain=False):
        plain_repr = '%s%s %s' % (self.type.__repr__(plain=True),
                                  Field.get_cardinality_repr(self.cardinality),
                                  self.name)
        if plain: return plain_repr
        else: return 'Field(%s)' % plain_repr

    @staticmethod
    def get_cardinality_repr(cardinality):
        return '' if cardinality == 'single' else '?' if cardinality == 'optional' else '*'


class ASDLType(object):
    def __init__(self, type_name):
        self.name = type_name

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return isinstance(other, ASDLType) and self.name == other.name

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self, plain=False):
        plain_repr = self.name
        if plain: return plain_repr
        else: return '%s(%s)' % (self.__class__.__name__, plain_repr)


class ASDLCompositeType(ASDLType):
    pass


class ASDLPrimitiveType(ASDLType):
    pass


if __name__ == '__main__':
    asdl_desc = """
var, ent, num, var_type

expr = Variable(var variable)
| Entity(ent entity)
| Number(num number)
| Apply(pred predicate, expr* arguments)
| Argmax(var variable, expr domain, expr body)
| Argmin(var variable, expr domain, expr body)
| Count(var variable, expr body)
| Exists(var variable, expr body)
| Lambda(var variable, var_type type, expr body)
| Max(var variable, expr body)
| Min(var variable, expr body)
| Sum(var variable, expr domain, expr body)
| The(var variable, expr body)
| Not(expr argument)
| And(expr* arguments)
| Or(expr* arguments)
| Compare(cmp_op op, expr left, expr right)

cmp_op = GreaterThan | Equal | LessThan
"""

    grammar = ASDLGrammar.from_text(asdl_desc)
    print(ASDLCompositeType('1') == ASDLPrimitiveType('1'))


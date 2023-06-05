# -*- coding: utf-8 -*-
import os, json, pickle, argparse, sys, time
import pdb

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from supar import Parser

MOD = ['nn', 'amod', 'advmod', 'rcmod', 'partmod', 'poss', 'neg', 'predet', 'acomp', 'advcl', 'ccomp', 'tmod',
                  'mark', 'xcomp', 'appos', 'npadvmod', 'infmod'] + \
                 ['num', 'number', 'quantmod'] + ['pobj', 'dobj', 'iobj']

def quote_normalization(question):
    """ Normalize all usage of quotation marks into a separate \" """
    new_question, quotation_marks = [], ["'", '"', '`', '‘', '’', '“', '”', '``', "''", "‘‘", "’’"]
    for idx, tok in enumerate(question):
        if len(tok) > 2 and tok[0] in quotation_marks and tok[-1] in quotation_marks:
            new_question += ["\"", tok[1:-1], "\""]
        elif len(tok) > 2 and tok[0] in quotation_marks:
            new_question += ["\"", tok[1:]]
        elif len(tok) > 2 and tok[-1] in quotation_marks:
            new_question += [tok[:-1], "\"" ]
        elif tok in quotation_marks:
            new_question.append("\"")
        elif len(tok) == 2 and tok[0] in quotation_marks:
            # special case: the length of entity value is 1
            if idx + 1 < len(question) and question[idx + 1] in quotation_marks:
                new_question += ["\"", tok[1]]
            else:
                new_question.append(tok)
        else:
            new_question.append(tok)
    return new_question

def inject_syntax_dataset(processor, dataset, output_path=None):
    syntax_dataset = []
    for idx, data in enumerate(dataset):
        entry = processor.inject_syntax(data)
        syntax_dataset.append(entry)

        if idx % 100 == 0:
            print("************************ processing {}th dataset ************************".format(idx))
    if output_path:
        pickle.dump(syntax_dataset, open(output_path, "wb"))
    return syntax_dataset

def inject_syntax_dataset_json(processor, dataset, mode='train', output_path=None):
    syntax_dataset = []
    for idx, data in enumerate(dataset):
        entry = processor.inject_syntax(data)
        if mode == 'dev':
            # please switch the length of your training data.
            entry['graph_idx'] = idx + 8577 
        else:
            entry['graph_idx'] = idx
        syntax_dataset.append(entry)

        if idx % 1000 == 0:
            print("************************ processing {}th dataset ************************".format(idx))
    if output_path:
        json.dump(syntax_dataset, open(output_path, "w"), indent=4)
    return syntax_dataset

class DEP():

    def __init__(self, parser):
        super(DEP, self).__init__()

        self.parser = parser

    def acquire_dep(self, entry):
        dep_dict = {}
        question = ' '.join(quote_normalization(entry['question_toks']))
        parsed_question = self.parser.predict(question, lang='en', prob=False, verbose=False)
        arcs = [i - 1 for i in parsed_question.arcs[0]]
        rels = parsed_question.rels[0]
        # construct a dict:
        for tgt, src in enumerate(arcs):
            rel = rels[tgt]
            if rel in MOD:
                rel = 'mod'
            else:rel = 'arg'
            dep_dict[tgt] = [src, rel]

        return dep_dict

    def inject_syntax(self, entry):
        question = ' '.join(quote_normalization(entry['question_toks']))
        relation_matrix = entry['relations']
        ori_question = entry['processed_question_toks']
        # inject relations:
        parsed_question = self.parser.predict(question, lang='en', prob=False, verbose=False)
        arcs = [i - 1 for i in parsed_question.arcs[0]]
        rels = parsed_question.rels[0]
        # construct
        if len(relation_matrix) != len(arcs):
            print("mismatched: {}".format(question))
            print("processed: {}".format(ori_question))
        for tgt, src in enumerate(arcs):

            if src < 0:
                continue
            rel = rels[tgt]
            if rel in MOD:
                relation_matrix[src][tgt] = 'question-question-modifier'
            else:
                relation_matrix[src][tgt] = 'question-question-argument'
        entry['relations'] = relation_matrix
        return entry


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--dataset_path', type=str, required=False,default = "data/train.bin", help='dataset path')
    arg_parser.add_argument('--database_path', type=str, required=False, help='database path')
    arg_parser.add_argument('--mode', type=str, required=False, default = "train", help='train or dev')
    arg_parser.add_argument('--output_path', type=str, help='output path', default = "data/syntax.bin")
    # arg_parser.add_argument('--raw_table_path', type=str, required=False, help='raw table path')
    # arg_parser.add_argument('--plm_path', type=str, required=True, help='plm path')
    args = arg_parser.parse_args()

    # load demo data:
    dataset = pickle.load(open(args.dataset_path, "rb"))
    # parser = Parser.load('biaffine-dep-en')
    parser = Parser.load(path='data_all_in/preprocess/ptb.biaffine.dep.lstm.char')
    
    print("successfully load parser")
    

    dep = DEP(parser=parser)


    syntax_dataset = inject_syntax_dataset_json(dep, dataset=dataset, mode=args.mode, output_path=args.output_path)


    print("successfully processed to {}".format(str(args.output_path)))
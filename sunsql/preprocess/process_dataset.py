#coding=utf8
import os, json, pickle, argparse, sys, time
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from asdl.asdl import ASDLGrammar
from asdl.transition_system import TransitionSystem
from asdl.action_info import get_action_infos
from preprocess.common_utils import Preprocessor

def process_example(processor, entry, db, trans, verbose=False):
    # preprocess raw tokens, schema linking and subgraph extraction
    entry = processor.pipeline(entry, db, verbose=verbose)
    # generate target output actions
    ast = trans.surface_code_to_ast(entry['sql'])
    actions = trans.get_actions(ast)
    entry['ast'] = ast
    entry['actions'] = get_action_infos(tgt_actions=actions)
    return entry

def process_tables(processor, tables_list, output_path=None, verbose=False):
    tables = {}
    for each in tables_list:
        if verbose:
            print('*************** Processing database %s **************' % (each['db_id'])) #
        tables[each['db_id']] = processor.preprocess_database(each, verbose=verbose)
    print('In total, process %d databases .' % (len(tables)))
    if output_path is not None:
        pickle.dump(tables, open(output_path, 'wb'))
    return tables

def process_dataset(processor, dataset, tables, output_path=None, skip_large=False, verbose=False):
    from utils.constants import GRAMMAR_FILEPATH
    grammar = ASDLGrammar.from_filepath(GRAMMAR_FILEPATH)
    trans = TransitionSystem.get_class_by_lang('sql')(grammar)
    processed_dataset = []
    cnt=0
    for idx, entry in enumerate(dataset):
        if skip_large and len(tables[entry['db_id']]['column_names']) > 100: continue
        if verbose:
            print('*************** Processing %d-th sample **************' % (idx))
        entry = process_example(processor, entry, tables[entry['db_id']], trans, verbose=verbose)
        # if cnt%2==1:
        #     # print('here')
        #     assert entry['query'] == processed_dataset[-1]['query']
        cnt+=1
        processed_dataset.append(entry)
    print('In total, process %d samples , skip %d extremely large databases.' % (len(processed_dataset), len(dataset) - len(processed_dataset)))
    if output_path is not None:
        # serialize preprocessed dataset
        pickle.dump(processed_dataset, open(output_path, 'wb'))
    return processed_dataset

if __name__ == '__main__':

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--db_dir', type=str, default='data/database')
    arg_parser.add_argument('--dataset_path', type=str, required=True, help='dataset path')
    arg_parser.add_argument('--raw_table_path', type=str, help='raw tables path')
    arg_parser.add_argument('--table_path', type=str, required=True, help='processed table path')
    arg_parser.add_argument('--output_path', type=str, required=True, help='output preprocessed dataset')
    arg_parser.add_argument('--skip_large', action='store_true', help='whether skip large databases')
    arg_parser.add_argument('--verbose', action='store_true', help='whether print processing information')
    args = arg_parser.parse_args()

    processor = Preprocessor(db_dir=args.db_dir, db_content=True)
    # loading database and dataset
    if args.raw_table_path:
        # need to preprocess database items
        tables_list = json.load(open(args.raw_table_path, 'r'))
        print('Firstly, preprocess the original databases ...')
        start_time = time.time()
        tables = process_tables(processor, tables_list, args.table_path, args.verbose)
        print('Databases preprocessing costs %.4fs .' % (time.time() - start_time))
    else:
        tables = pickle.load(open(args.table_path, 'rb'))
    dataset = json.load(open(args.dataset_path, 'r'))
    start_time = time.time()
    dataset = process_dataset(processor, dataset, tables, args.output_path, args.skip_large, verbose=args.verbose)
    print('Dataset preprocessing costs %.4fs .' % (time.time() - start_time))

import json
import argparse

def merge_train(train_spider, train_others, output_path=None):
    total_train = train_spider + train_others
    
    if output_path:
        json.dump(total_train, open(output_path, "w"), indent=4)

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--train_spider_path', type=str, required=True, help='train_spider_path')
    arg_parser.add_argument('--train_others_path', type=str, required=True, help='train_others_path')
    arg_parser.add_argument('--train_output_path', type=str, required=False, help='train_output_path')
    args = arg_parser.parse_args()

    train_spider = json.load(open(args.train_spider_path, 'r'))
    train_others = json.load(open(args.train_others_path, 'r'))
    
    merge_train(train_spider, train_others, output_path=args.train_output_path)
    print("merged training data")
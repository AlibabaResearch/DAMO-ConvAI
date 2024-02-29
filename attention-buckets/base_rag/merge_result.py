import os
import json
import argparse
import regex
import unicodedata
import string


def normalize_answer(s):
    def remove_articles(text):
        return regex.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(lower(s)))



def main(args):
    data_file = args.data_dir + args.data_name +'-test.jsonl'
    # answer_file = args.answer_file #args.result_dir + args.data_name + '_doc_num' + str(args.num_doc)
    answer_file = 'answer/nq_doc_num10(10000, 13000, 16000, 19000, 22000, 25000, 28000)'
    base_list_file =  answer_file +'/base_list.json'  
    if args.ngpu > 1:
        f2 = open(base_list_file, 'w')
        for i in range(args.ngpu):
            base_list_file_tmp =  answer_file +'/base_list'+'_'+str(i) + '.json' 
            f2_tmp = open(base_list_file_tmp).read()
            f2.write(f2_tmp)
        f2.close()
        
    # base_list_file = 'answer/webq_doc_num10/base_(10000,17000,18000,19000,20000,23000,25000).json'
    f2 = open(base_list_file, 'r').readlines()
    total = len(f2)
    
    true = 0
    with open(data_file) as fin:
        data = json.load(fin)
        # assert total == len(data)
        for idx, input_example in enumerate(data):
            gold_answer = [x.strip() for x in input_example["answers"]]
            answer = json.loads(f2[idx])['answer']
            # print(answer)
            # print(gold_answer)
            

            for x in gold_answer: 
                if normalize_answer(x).lower() in normalize_answer(answer).lower():
                    true += 1
                    break


    print(true)
    print(total)
    print(true/total)
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', type=str, default='meta-llama/Llama-2-7b-chat-hf')
    parser.add_argument('--model_type', type=str, default='llama')
    parser.add_argument('--data_dir', type=str, default='../qa_dataset/')
    parser.add_argument('--prompt_dir', type=str, default='prompts/qa.prompt')
    parser.add_argument('--result_dir', type=str, default='answer/')
    parser.add_argument('--answer_file', type=str, default='answer/')
    parser.add_argument('--num_doc', type=int,  default=10)
    parser.add_argument('--bsz', type=int,  default=1)
    parser.add_argument('--data_name', type=str,  default='nq')
    parser.add_argument('--ngpu', type=int, default=1)
    parser.add_argument('--chosen_base', type=int, default=10000)
    args = parser.parse_args()

    main(args)

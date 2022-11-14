from curses import meta
import os
from os.path import join, exists
import json
from tqdm import tqdm
import copy
from rich import print


def load_file():
    split_list = ['dev', 'train', 'devtest']
    output = []
    
    for split in split_list:
        
        file_path = f'../data_dstc11/simmc2.1_dials_dstc11_{split}.json'
        with open(file_path) as f_in:
            data = json.load(f_in)['dialogue_data']

        for dialogue in data:
            line_output = []
            dialogue_idx = dialogue['dialogue_idx']
            scene_ids = list(sorted(dialogue['scene_ids'].items(), key=lambda item: int(item[0])))
            domain = dialogue['domain']
            
            history_object = []
            prev_turn = None
            
            for iter_idx, turn in enumerate(dialogue['dialogue']):
                turn_idx = turn['turn_idx']
                transcript = turn['transcript']
                system_transcript = turn['system_transcript']
                
                user_slot_values = turn['transcript_annotated']['act_attributes']['slot_values']
                user_objects = turn['transcript_annotated']['act_attributes']['objects']
                
                system_slot_values = turn['system_transcript_annotated']['act_attributes']['slot_values']
                
                disambiguation_label = turn['transcript_annotated']['disambiguation_label']
                disambiguation_candidates = turn['transcript_annotated']['disambiguation_candidates']
                disambiguation_candidates_raw = turn['transcript_annotated']['disambiguation_candidates_raw']
            
                if disambiguation_label == 1:
                    disambiguation_transcript = turn.get('disambiguation_transcript', 'None')
                    output.append({
                        'disambiguation_transcript': disambiguation_transcript,
                        'transcript': transcript,
                        'disambiguation_candidates': disambiguation_candidates,
                        'disambiguation_candidates_raw': disambiguation_candidates_raw,
                        'split': split,
                        'type': ''
                    })

                prev_turn = turn
        
        with open('./result/output.json', 'w') as f_out:
            json.dump(output, f_out, indent=4, ensure_ascii=False)


def main():
    load_file()


if __name__ == '__main__':
    main()
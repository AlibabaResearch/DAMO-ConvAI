import json
KEYWORDS =  json.load(open("data/preprocessed_data/bart_parser_label_mapping_2.json"))["keyword"]
# KEYWORDS = json.load(open("data/preprocessed_data/bart_parser_pretrain_label_mapping.json"))["keyword"]
SKETCH_KEYWORDS = json.load(open("data/preprocessed_data/bart_parser_label_mapping.json"))["keyword"]

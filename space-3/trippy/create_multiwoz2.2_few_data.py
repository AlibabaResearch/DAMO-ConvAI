import json


def create_examples(input_file, output_file, mode):
    examples = {}
    ids = []

    with open(f"data/MULTIWOZ2.2/{mode}_dials.json", "r", encoding='utf-8') as reader:
        all_examples = json.load(reader)

    if mode == 'train':
        with open(input_file, "r", encoding='utf-8') as reader:
            input_data = json.load(reader)
        for id in input_data:
            if id not in ids:
                ids.append(id)
        for dialog_id, dialog in all_examples.items():
            if dialog_id in ids:
                examples[dialog_id] = dialog
        assert len(examples) == len(input_data)
    else:
        examples = all_examples

    with open(output_file, "w") as f:
        json.dump(examples, f, indent=2, separators=(",", ": "), sort_keys=True)
    print(f'{mode}: saved {len(examples)} samples to {output_file}')


if __name__ == '__main__':
    data_name = 'MULTIWOZ2.2_fewshot'
    for mode in ["train", "val", "test"]:
        input_file = f'data/MULTIWOZ2.1_fewshot/{mode}_dials.json'
        output_file = f"data/{data_name}/{mode}_dials.json"
        create_examples(input_file=input_file, output_file=output_file, mode=mode)

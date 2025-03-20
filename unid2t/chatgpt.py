import openai

import argparse
from tqdm import tqdm

from data_preprocess import utils
# # OpenAI API Key
openai.api_key = ""


def convert_one_example(example, is_simplified_data=True, is_lower_relation=True):
    """

    :param example: dict{
        tripleset: [
            [head, relation, tail]
        ],
        subtree_was_extended: bool
        annotations: [dict{
            source: str,
            text: str
        }]
    }
    :return: converted_example: dict{
        tripleset: [
            [head, relation, tail]
        ],
        subtree_was_extended: bool

        linear_node: [str]
        triple: list([head_id, tail_id])
        metadata: []
        annotations_source: [source: str]
        target_sents: [text: str]


    }
    :param is_simplified_data
    :param is_lower_relation
    """
    tripleset = example['tripleset']
    Structured_Input = ""
    for i,triple in enumerate(tripleset):
        for j,t in enumerate(triple):
            Structured_Input += t
            if j != len(triple) - 1:
                Structured_Input += " : "
        if i != len(tripleset) - 1:
            Structured_Input += " | "


    return Structured_Input



def convert(input_file_src, output_file_src):

    dataset = utils.read_json_file(input_file_src)

    converted_dataset = []
    print("!1")
    for i, example in tqdm(enumerate(dataset)):
        converted_example = convert_one_example(example)
        prompt = "Put the triples together to form a sentence: "
        prompt = prompt + converted_example
        utils.write_to_json_file_add(prompt, output_file_src)
        try:
            response = openai.Completion.create(
              model="text-davinci-003",
                prompt=prompt,
              # prompt="Put the triples together to form a sentence: Hawaii Five-O, NOTES, Episode: The Flight of the Jewels, [TABLECONTEXT], [TITLE], Jeff Daniels, [TABLECONTEXT], TITLE, Hawaii Five-O",
                temperature=0,
                max_tokens=256,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0
            )
            converted_dataset.append(response["choices"][0]["text"])
            utils.write_to_json_file_add(prompt, output_file_src)
        except:
            utils.write_to_json_file_add(i, output_file_src)





    # utils.write_to_json_file_by_line(converted_dataset, output_file_src)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert wikitablet to unified graph')

    parser.add_argument("--input_file_src", type=str,
                        default='/dart/data/v1.1.1/dart-v1.1.1-full-test.json')
    parser.add_argument("--output_file_src", type=str,
                        default='/chatgpt/dart_test.json')
    args = parser.parse_args()

    convert(input_file_src=args.input_file_src, output_file_src=args.output_file_src)


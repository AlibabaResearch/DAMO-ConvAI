import sys
sys.path.append('../data_process')
from typing import List, Optional, Tuple
# from QAInput import StructuralQAInput as QAInput
#from QAInput import QAInputNoPrompt as QAInput

def preprocess_all(
        examples,
        question_column: str,
        answer_column: str,
)-> Tuple[List[str], List[str]]:
    questions = examples[question_column]
    answers = examples[answer_column]
    inputs = questions
    targets = answers
    return inputs, targets
        

def preprocess_sqaud_batch(
        examples,
        question_column: str,
        context_column: str,
        answer_column: str,
) -> Tuple[List[str], List[str]]:
    questions = examples[question_column]
    contexts = examples[context_column]
    answers = examples[answer_column]
    inputs = [QAInput.qg_input_extractive_qa(context, question) for question, context in zip(questions, contexts)]
    targets = [answer["text"][0] if len(answer["text"]) > 0 else "" for answer in answers]
    return inputs, targets

def preprocess_sqaud_abstractive_batch(
        examples,
        question_column: str,
        context_column: str,
        answer_column: str,
) -> Tuple[List[str], List[str]]:
    questions = examples[question_column]
    contexts = examples[context_column]
    answers = examples[answer_column]
    inputs = [QAInput.qg_input_abstrativeqa(context, question) for question, context in zip(questions, contexts)]
    targets = [answer["text"][0] if len(answer["text"]) > 0 else "" for answer in answers]
    return inputs, targets

def preprocess_boolq_batch(
        examples,
        question_column: str,
        context_column: str,
        answer_column: str,
) -> Tuple[List[str], List[str]]:
    question_column, context_column, answer_column = 'question', 'passage', 'answer'
    questions = examples[question_column]
    contexts = examples[context_column]
    answers = examples[answer_column]
    inputs = [QAInput.qg_input_boolqa(context, question) for question, context in zip(questions, contexts)]
    targets = [str(ans) for ans in answers] #[answer["text"][0] if len(answer["text"]) > 0 else "" for answer in answers]
    # print(inputs,targets)
    return inputs, targets

def preprocess_boolq_batch_pretrain(
        examples,
        question_column: str,
        context_column: str,
        answer_column: str,
) -> Tuple[List[str], List[str]]:
    questions = examples[question_column]
    contexts = examples[context_column]
    answers = examples[answer_column]
    inputs = [QAInput.qg_input_boolqa(context, question) for question, context in zip(questions, contexts)]
    targets = [answer["text"][0].capitalize() if len(answer["text"]) > 0 else "" for answer in answers]
    # print(inputs,targets)
    return inputs, targets

def preprocess_narrativeqa_batch(
        examples,
        question_column: str,
        context_column: str,
        answer_column: str,
) -> Tuple[List[str], List[str]]:
    contexts = [exp['summary']['text'] for exp in examples['document']]
    questions = [exp['text'] for exp in examples['question']]
    answers = [ans[0]['text'] for ans in examples['answers']]
    inputs = [QAInput.qg_input_abstrativeqa(context, question) for question, context in zip(questions, contexts)]
    targets = answers #[answer["text"][0] if len(answer["text"]) > 0 else "" for answer in answers]
    return inputs, targets

def preprocess_narrativeqa_batch_pretrain(
        examples,
        question_column: str,
        context_column: str,
        answer_column: str,
) -> Tuple[List[str], List[str]]:
    questions = examples[question_column]
    contexts = examples[context_column]
    answers = examples[answer_column]
    inputs = [QAInput.qg_input_abstrativeqa(context, question) for question, context in zip(questions, contexts)]
    targets = [answer["text"][0] if len(answer["text"]) > 0 else "" for answer in answers]
    return inputs, targets

def preprocess_drop_batch(
        examples,
        question_column: str,
        context_column: str,
        answer_column: str,
) -> Tuple[List[str], List[str]]:
    contexts = examples['passage']
    questions = examples['question']
    answers = examples['answers']     
    inputs = [QAInput.qg_input_abstrativeqa(context, question) for question, context in zip(questions, contexts)]
    targets = [answer["text"][0] if len(answer["text"]) > 0 else "" for answer in answers]
    return inputs, targets

def preprocess_race_batch(
        examples,
        question_column: str,
        context_column: str,
        answer_column: str,
) -> Tuple[List[str], List[str]]:
    contexts = examples['article']
    questions = examples['question']
    all_options = examples['options']
    answers = examples['answer']
    options_texts = [f'options: A. {options[0]}; B. {options[1]}; C. {options[2]}; D. {options[3]}' for options in all_options]
    inputs = [QAInput.qg_input_multirc(context, question, ops) for question, context, ops in zip(questions, contexts, options_texts)]
    ans_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3 }
    targets = [options[ans_map[answer]] for options, answer in zip(all_options, answers)]
    return inputs, targets

def preprocess_newsqa_batch(
        examples,
        question_column: str,
        context_column: str,
        answer_column: str,
) -> Tuple[List[str], List[str]]:
    questions = examples[question_column]
    contexts = examples[context_column]
    answers = examples[answer_column]
    inputs = [QAInput.qg_input_extractive_qa(context, question) for question, context in zip(questions, contexts)]
    # inputs = [QAInput.qg_input_abstrativeqa(context, question) for question, context in zip(questions, contexts)]
    targets = [answer["text"][0] if len(answer["text"]) > 0 else "" for answer in answers]
    return inputs, targets


def preprocess_ropes_batch(
        examples,
        question_column: str,
        context_column: str,
        answer_column: str,
) -> Tuple[List[str], List[str]]:
    questions = examples[question_column]
    backgrounds = examples["background"]
    situations = examples["situation"]
    answers = examples[answer_column]

    inputs = [QAInput.qg_input_extractive_qa(" ".join([background, situation]), question) for question, background, situation in zip(questions, backgrounds, situations)]
    targets = [answer["text"][0] if len(answer["text"]) > 0 else "" for answer in answers]
    return inputs, targets

def preprocess_openbookqa_batch(
        examples,
        question_column: str,
        context_column: str,
        answer_column: str,
) -> Tuple[List[str], List[str]]:
    questions = examples['question_stem']
    all_options = examples['choices']
    answers = examples['answerKey']
    options_texts = [f"options: A. {options['text'][0]}; B. {options['text'][1]}; C. {options['text'][2]}; D. {options['text'][3]}" for options in all_options]
    inputs = [QAInput.qg_input_multirc("", question, ops) for question, ops in zip(questions, options_texts)]
    ans_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    targets = [options['text'][ans_map[answer]] for options, answer in zip(all_options, answers)]
    return inputs, targets

def preprocess_social_iqa_batch(
        examples,
        question_column: str,
        context_column: str,
        answer_column: str,
) -> Tuple[List[str], List[str]]:
    contexts = examples['article']
    questions = examples['question']
    all_options = examples['options']
    answers = examples['answer']
    options_texts = [f'options: A. {options[0]}; B. {options[1]}; C. {options[2]}' for options in all_options]
    inputs = [QAInput.qg_input_multirc(context, question, ops) for question, context, ops in zip(questions, contexts, options_texts)]
    ans_map = {'A': 0, 'B': 1, 'C': 2,}
    targets = [options[ans_map[answer]] for options, answer in zip(all_options, answers)]
    return inputs, targets

def preprocess_dream_batch(
        examples,
        question_column: str,
        context_column: str,
        answer_column: str,
) -> Tuple[List[str], List[str]]:
    contexts = [" ".join(dialogue) for dialogue in examples['dialogue']]
    questions = examples['question']
    all_options = examples['choice']
    answers = examples['answer']
    answer_idxs = [options.index(answer) for answer, options in zip(answers, all_options)]
    options_texts = [f'options: A. {options[0]}; B. {options[1]}; C. {options[2]}' for options in all_options]
    inputs = [QAInput.qg_input_multirc(context, question, ops) for question, context, ops in zip(questions, contexts, options_texts)]
    targets = answers
    return inputs, targets
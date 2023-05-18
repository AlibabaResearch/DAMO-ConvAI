            
class QAInput:
    @classmethod
    def qg_input_abstrativeqa(cls, context, question,hint, options=None):
        question = question
        source_text = f'[TASK] [ABSTRACTIVE] [QUESTION] {question}. [CONTEXT] {context} the answer is: {hint}'
        return source_text

    @classmethod
    def qg_input_boolqa(cls, context, question, hint,options=None):
        source_text = f'[TASK] [BOOL] [QUESTION] {question}. [CONTEXT] {context} the answer is: {hint}'
        return source_text

    @classmethod
    def qg_input_extractive_qa(cls, context, question, hint,options=None):
        source_text = f'[TASK] [EXTRACTIVE] [QUESTION] {question.lower().capitalize()}. [CONTEXT] {context} the answer is: {hint}'
        return source_text

    @classmethod
    def qg_input_multirc(cls, context, question,hint, options=None):
        source_text = f'[TASK] [MultiChoice] [QUESTION] {question} [OPTIONS] {options} [CONTEXT] {context} the answer is: {hint}'
        return source_text

def preprocess_proqa_eval(
        examples,
        question_column: str,
        answer_column:str,
        hint_column:str,
        format_name:str):
    questions = examples[question_column]
    answers = examples[answer_column]
    hints = examples[hint_column]

  #  single_inputs = []
    if format_name=="extractive":
        inputs = [QAInput.qg_input_extractive_qa(question.split("\\n")[1], question.split("\\n")[0],hint='') for hint,question in zip(hints,questions)]

    if format_name=="abstractive":
        inputs = [QAInput.qg_input_abstrativeqa(question.split("\\n")[1], question.split("\\n")[0],hint='') for hint,question in zip(hints,questions)]

    if format_name=="multichoice":
        if len(questions[0].split("\\n"))==2:
            inputs = [QAInput.qg_input_multirc(context = '',question = question.split("\\n")[0], options=question.split("\\n")[1],hint='') for hint,question in zip(hints,questions)]

        else:
            inputs = [QAInput.qg_input_multirc(context = question.split("\\n")[2],question = question.split("\\n")[0], options=question.split("\\n")[1],hint='') for hint,question in zip(hints,questions)]

    if format_name =="bool":
        inputs = [QAInput.qg_input_boolqa(context = question.split("\\n")[1],question = question.split("\\n")[0],hint='') for hint,question in zip(hints,questions)]


    targets = answers
    return inputs,targets,examples[hint_column]
    
    
def preprocess_proqa(
        examples,
        question_column: str,
        answer_column:str,
        hint_column:str,
        format_name:str):
    questions = examples[question_column]
    answers = examples[answer_column]
    hints = examples[hint_column]
  #  single_inputs = []
    if format_name=="extractive":
        inputs = [QAInput.qg_input_extractive_qa(question.split("\\n")[1], question.split("\\n")[0],"") for hint,question in zip(hints,questions)]
    if format_name=="abstractive":
        inputs = [QAInput.qg_input_abstrativeqa(question.split("\\n")[1], question.split("\\n")[0],"") for hint,question in zip(hints,questions)]
    if format_name=="multichoice":
        if len(questions[0].split("\\n"))==2:
            inputs = [QAInput.qg_input_multirc(context = '',question = question.split("\\n")[0], options=question.split("\\n")[1],hint="") for hint,question in zip(hints,questions)]
        else:
            inputs = [QAInput.qg_input_multirc(context = question.split("\\n")[2],question = question.split("\\n")[0], options=question.split("\\n")[1],hint="") for hint,question in zip(hints,questions)]
    if format_name =="bool":
        inputs = [QAInput.qg_input_boolqa(context = question.split("\\n")[1],question = question.split("\\n")[0],hint="") for hint,question in zip(hints,questions)]


    targets = answers
    return inputs, targets, examples[hint_column]

def preprocess_simple(
        examples,
        question_column: str,
        answer_column:str,
        format_name:str):
    questions = examples[question_column]
    answers = examples[answer_column]
  #  single_inputs = []
    if format_name=="extractive":
        inputs = [QAInput.qg_input_extractive_qa(question.split("\\n")[1], question.split("\\n")[0],"") for question in questions]
    if format_name=="abstractive":
        inputs = [QAInput.qg_input_abstrativeqa(question.split("\\n")[1], question.split("\\n")[0],"") for question in questions]
    if format_name=="multichoice":
        if len(questions[0].split("\\n"))==2:
            inputs = [QAInput.qg_input_multirc(context = '',question = question.split("\\n")[0], options=question.split("\\n")[1],hint="") for question in questions]
        else:
            inputs = [QAInput.qg_input_multirc(context = question.split("\\n")[2],question = question.split("\\n")[0], options=question.split("\\n")[1],hint="") for question in questions]
    if format_name =="bool":
        inputs = [QAInput.qg_input_boolqa(context = question.split("\\n")[1],question = question.split("\\n")[0],hint="") for question in questions]


    targets = answers
    return inputs, targets
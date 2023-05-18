#ProQA
#[paper]https://arxiv.org/abs/2205.04040
class StructuralQAInput:
    @classmethod
    def qg_input(cls, context, question, options=None):
        question = question
        source_text = f'[QUESTION] {question}. [CONTEXT] {context} the answer is: '
        return source_text






class  SimpleQAInput:
    @classmethod
    def qg_input(cls, context, question, options=None):
        question = question
        source_text = f'{question} \\n {context}'
        return source_text


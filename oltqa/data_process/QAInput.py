class StructuralQAInput:
    @classmethod
    def qg_input_abstrativeqa(cls, context, question, options=None):
        question = question
        source_text = f'[TASK] [ABSTRACTIVE] [QUESTION] {question}. [CONTEXT] {context} the answer is: '
        return source_text

    @classmethod
    def qg_input_boolqa(cls, context, question, options=None):
        source_text = f'[TASK] [BOOL] [QUESTION] {question}. [CONTEXT] {context} the answer is: '
        return source_text

    @classmethod
    def qg_input_extractive_qa(cls, context, question, options=None):
        source_text = f'[TASK] [EXTRACTIVE] [QUESTION] {question.lower().capitalize()}. [CONTEXT] {context} the answer is: '
        return source_text

    @classmethod
    def qg_input_multirc(cls, context, question, options=None):
        source_text = f'[TASK] [MultiChoice] [QUESTION] {question} [OPTIONS] {options} [CONTEXT] {context} the answer is: '
        return source_text


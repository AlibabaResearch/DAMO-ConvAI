class QGInput:
    @classmethod
    def qg_input_abstractiveqa(cls,context,question,answers,options=None):
        outputs = []
        for ans in answers:
            source_text = f'The context: {context}, the answer is: {ans}. This is a summary task, please generate question: '
            target_text = f'{question}'
            outputs.append({'source_text':source_text,'target_text':target_text})
        return outputs
    @classmethod
    def qg_input_abstractiveqa_qapairs(cls,context,question,answers,options=None):
        outputs = []
        for ans in answers:
            source_text = f'Generate question and the answer: The context: {context}.'
            target_text = f'[Question]: {question} [Answer] {ans}'
            outputs.append({'source_text': source_text, 'target_text': target_text})
        return outputs
    @classmethod
    def qg_input_boolqa(cls,context,question,answers,options=None):
        outputs = []
        for ans in answers:
            source_text = f'The context: {context}. The answer is: <hl> {ans} <hl>. This is a bool task, generate question: '
            target_text = f'{question}'
            outputs.append({'source_text':source_text,'target_text':target_text})
        return outputs
    @classmethod
    def qg_input_extractive_qa(cls, context, question, answers,options=None):
        outputs = []
        for ans in answers:
            context = context.replace(ans,f'<hl> {ans} <hl>')
            source_text = f'The context: {context}. The answer is: {ans}. This is a extractive task, generate question: '
            target_text = f'{question}'
            outputs.append({'source_text': source_text, 'target_text': target_text})
        return outputs

    @classmethod
    def qg_input_extractive_qapairs(cls, context, question, answers, options=None):
        outputs = []
        for ans in answers:
            source_text = f'Generate question and the answer: [Context] {context}.'
            target_text = f'[question] {question} [answer] {ans}'
            outputs.append({'source_text': source_text, 'target_text': target_text})
        return outputs
    @classmethod
    def qg_input_multirc(cls, context, question, answers, options=None):
        outputs = []
        for ans in answers:
            source_text = f'The context: {context}. The options are: {options}. The answer is: <hl> {ans} <hl>. This is a machine reading comprehension task, generate question: '
            target_text = f'{question}'
            outputs.append({'source_text': source_text, 'target_text': target_text})
        return outputs
    @classmethod
    def qg_intput_multirc_negoption(cls, context, question, answers, options=None):
        outputs = []
        for ans in answers:
            for option in options:
                if option!=ans:
                    source_text = f'Context: {context}. Answer is: <hl> {ans} <hl>. Generate false option: '
                    target_text = f'{option}'
                    outputs.append({'source_text': source_text, 'target_text': target_text})
        return outputs
    @classmethod
    def qg_intput_multirc_qapairs(cls, context, question, answers, options=None):
        outputs = []
        for ans in answers:
                source_text = f'Generate question and answer: [Context] {context}'
                target_text = f'[Question] {question} [Answer] {ans}'
                outputs.append({'source_text': source_text, 'target_text': target_text})
        return outputs
    @classmethod
    def qg_intput_multirc_negoption_withq(cls, context, question, answers, options=None):
        outputs = []
        for ans in answers:
            for option in options:
                if option != ans:
                    source_text = f'Generate false option: [Context] {context}. [Question] {question}. [Answer] <hl> {ans} <hl>.'
                    target_text = f'{option}'
                    outputs.append({'source_text': source_text, 'target_text': target_text})
        return outputs

    @classmethod
    def qg_intput_multirc_qfirst(cls, context, question, answers, options=None):
        outputs = []
        for ans in answers:
            source_text = f'Generate question: Context: {context}. Answer: <hl> {ans} <hl>. '
            target_text = f'{question}'
            outputs.append({'source_text': source_text, 'target_text': target_text})
        return outputs
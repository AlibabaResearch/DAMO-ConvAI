"""
Retrieval pool of candidates
"""
from dataclasses import dataclass
from typing import List, Dict
import json


class OpenAIQARetrievePool(object):
    def __init__(self, data_path):
        with open(data_path, 'r') as f:
            data = json.load(f)
        self.data = []
        for d in data:
            if isinstance(d['qa_column'], List):
                d['qa_column'] = '|'.join(d['qa_column'])
            qa_item = QAItem(
                id=d['id'],
                qa_question=d['qa_question'],
                qa_column=d['qa_column'],
                qa_answer=d['qa_answer'],
                table=d['table'],
                title=d['title']
            )
            self.data.append(qa_item)

        self.pointer = 0

    def __iter__(self):
        return self

    def __next__(self):
        pointer = self.pointer
        if pointer < len(self):
            self.pointer += 1
            return self.data[pointer]
        else:
            self.pointer = 0
            raise StopIteration

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


@dataclass
class QAItem(object):
    id: int = None
    qa_question: str = None
    qa_column: str = None
    qa_answer: str = None
    table: Dict = None
    title: str = None
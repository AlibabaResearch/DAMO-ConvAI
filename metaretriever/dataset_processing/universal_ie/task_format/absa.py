#!/usr/bin/env python
# -*- coding:utf-8 -*-


import json
from typing import List
from universal_ie.utils import tokens_to_str, change_ptb_token_back
from universal_ie.ie_format import Entity, Label, Relation, Sentence, Span
from universal_ie.task_format.task_format import TaskFormat


class ABSA(TaskFormat):
    """ Aspect-Based Sentiment Analysis Data format at https://github.com/yhcc/BARTABSA."""

    def __init__(self, sentence_json, language='en'):
        super().__init__(
            language=language
        )
        self.tokens = sentence_json['words']
        for index in range(len(self.tokens)):
            self.tokens[index] = change_ptb_token_back(self.tokens[index])
        if self.tokens is None:
            print('[sentence without tokens]:', sentence_json)
            exit(1)
        self.aspects = sentence_json['aspects']
        self.opinions = sentence_json['opinions']

    def generate_instance(self):
        entities = dict()
        relations = list()
        entity_map = dict()

        for aspect, opinion in zip(self.aspects, self.opinions):
            aspect_span = (aspect['from'], aspect['to'])
            opinion_span = (opinion['from'], opinion['to'])

            if aspect_span not in entity_map:
                tokens = self.tokens[aspect_span[0]:aspect_span[1]]
                entities[aspect_span] = Entity(
                    span=Span(
                        tokens=tokens,
                        indexes=list(range(aspect_span[0], aspect_span[1])),
                        text=tokens_to_str(tokens, language=self.language),
                    ),
                    label=Label('aspect')
                )

            if opinion_span not in entity_map:
                tokens = self.tokens[opinion_span[0]:opinion_span[1]]
                entities[opinion_span] = Entity(
                    span=Span(
                        tokens=tokens,
                        indexes=list(range(opinion_span[0], opinion_span[1])),
                        text=tokens_to_str(tokens, language=self.language),
                    ),
                    label=Label('opinion')
                )

            relations += [Relation(
                arg1=entities[aspect_span],
                arg2=entities[opinion_span],
                label=Label(aspect['polarity']),
            )]

        return Sentence(
            tokens=self.tokens,
            entities=entities.values(),
            relations=relations,
        )

    @staticmethod
    def load_from_file(filename, language='en') -> List[Sentence]:
        sentence_list = list()
        raw_instance_list = json.load(open(filename))
        print(f"{filename}: {len(raw_instance_list)}")
        for instance in raw_instance_list:
            instance = ABSA(
                    sentence_json=instance,
                    language=language
                ).generate_instance()
            sentence_list += [instance]
        return sentence_list

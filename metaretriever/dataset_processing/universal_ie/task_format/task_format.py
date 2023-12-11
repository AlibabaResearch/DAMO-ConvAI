#!/usr/bin/env python
# -*- coding:utf-8 -*-
import abc


class TaskFormat:
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self, language='en'):
        self.language = language

    @abc.abstractmethod
    def generate_instance(self):
        pass

    @staticmethod
    @abc.abstractmethod
    def load_from_file(filename, language='en'):
        pass

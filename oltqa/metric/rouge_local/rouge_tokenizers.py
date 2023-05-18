# coding=utf-8
# Copyright 2022 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
"""Library containing Tokenizer definitions.
The RougeScorer class can be instantiated with the tokenizers defined here. New
tokenizers can be defined by creating a subclass of the Tokenizer abstract class
and overriding the tokenize() method.
"""
import abc
from nltk.stem import porter


import re
import six


# Pre-compile regexes that are use often
NON_ALPHANUM_PATTERN = r"[^a-z0-9]+"
NON_ALPHANUM_RE = re.compile(NON_ALPHANUM_PATTERN)
SPACES_PATTERN = r"\s+"
SPACES_RE = re.compile(SPACES_PATTERN)
VALID_TOKEN_PATTERN = r"^[a-z0-9]+$"
VALID_TOKEN_RE = re.compile(VALID_TOKEN_PATTERN)


def _tokenize(text, stemmer):
  """Tokenize input text into a list of tokens.
  This approach aims to replicate the approach taken by Chin-Yew Lin in
  the original ROUGE implementation.
  Args:
    text: A text blob to tokenize.
    stemmer: An optional stemmer.
  Returns:
    A list of string tokens extracted from input text.
  """

  # Convert everything to lowercase.
  text = text.lower()
  # Replace any non-alpha-numeric characters with spaces.
  text = NON_ALPHANUM_RE.sub(" ", six.ensure_str(text))

  tokens = SPACES_RE.split(text)
  if stemmer:
    # Only stem words more than 3 characters long.
    tokens = [six.ensure_str(stemmer.stem(x)) if len(x) > 3 else x
              for x in tokens]

  # One final check to drop any empty or invalid tokens.
  tokens = [x for x in tokens if VALID_TOKEN_RE.match(x)]

  return tokens


class Tokenizer(abc.ABC):
  """Abstract base class for a tokenizer.
  Subclasses of Tokenizer must implement the tokenize() method.
  """

  @abc.abstractmethod
  def tokenize(self, text):
    raise NotImplementedError("Tokenizer must override tokenize() method")


class DefaultTokenizer(Tokenizer):
  """Default tokenizer which tokenizes on whitespace."""

  def __init__(self, use_stemmer=False):
    """Constructor for DefaultTokenizer.
    Args:
      use_stemmer: boolean, indicating whether Porter stemmer should be used to
      strip word suffixes to improve matching.
    """
    self._stemmer = porter.PorterStemmer() if use_stemmer else None

  def tokenize(self, text):
    return _tokenize(text, self._stemmer)
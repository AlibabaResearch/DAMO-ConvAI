""" Class for the Sequence to sequence model for ATIS."""

import os

import torch
import torch.nn.functional as F
from . import torch_utils
from . import utils_bert

from data_util.vocabulary import DEL_TOK, UNK_TOK

from .encoder import Encoder
from .embedder import Embedder
from .token_predictor import construct_token_predictor

import numpy as np

from data_util.atis_vocab import ATISVocabulary

def get_token_indices(token, index_to_token):
    """ Maps from a gold token (string) to a list of indices.

    Inputs:
        token (string): String to look up.
        index_to_token (list of tokens): Ordered list of tokens.

    Returns:
        list of int, representing the indices of the token in the probability
            distribution.
    """
    if token in index_to_token:
        if len(set(index_to_token)) == len(index_to_token):  # no duplicates
            return [index_to_token.index(token)]
        else:
            indices = []
            for index, other_token in enumerate(index_to_token):
                if token == other_token:
                    indices.append(index)
            assert len(indices) == len(set(indices))
            return indices
    else:
        return [index_to_token.index(UNK_TOK)]

def flatten_utterances(utterances):
    """ Gets a flat sequence from a sequence of utterances.

    Inputs:
        utterances (list of list of str): Utterances to concatenate.

    Returns:
        list of str, representing the flattened sequence with separating
            delimiter tokens.
    """
    sequence = []
    for i, utterance in enumerate(utterances):
        sequence.extend(utterance)
        if i < len(utterances) - 1:
            sequence.append(DEL_TOK)

    return sequence

def encode_snippets_with_states(snippets, states):
    """ Encodes snippets by using previous query states instead.

    Inputs:
        snippets (list of Snippet): Input snippets.
        states (list of dy.Expression): Previous hidden states to use.
        TODO: should this by dy.Expression or vector values?
    """
    for snippet in snippets:
        snippet.set_embedding(torch.cat([states[snippet.startpos],states[snippet.endpos]], dim=0))
    return snippets

def load_word_embeddings(input_vocabulary, output_vocabulary, output_vocabulary_schema, params):
  print(output_vocabulary.inorder_tokens)
  print()

  def read_glove_embedding(embedding_filename, embedding_size):
    glove_embeddings = {}

    with open(embedding_filename) as f:
      cnt = 1
      for line in f:
        cnt += 1
        if params.debug or not params.train:
          if cnt == 1000:
            print('Read 1000 word embeddings')
            break
        l_split = line.split()
        word = " ".join(l_split[0:len(l_split) - embedding_size])
        embedding = np.array([float(val) for val in l_split[-embedding_size:]])
        glove_embeddings[word] = embedding

    return glove_embeddings
    
  print('Loading Glove Embedding from', params.embedding_filename)
  glove_embedding_size = 300
# glove_embeddings = read_glove_embedding(params.embedding_filename, glove_embedding_size)
  import pickle as pkl
#   pkl.dump(glove_embeddings, open("glove_embeddings.pkl", "wb"))
#   exit()
  glove_embeddings = pkl.load(open("glove_embeddings.pkl", "rb"))
  print('Done')

  input_embedding_size = glove_embedding_size

  def create_word_embeddings(vocab):
    vocabulary_embeddings = np.zeros((len(vocab), glove_embedding_size), dtype=np.float32)
    vocabulary_tokens = vocab.inorder_tokens

    glove_oov = 0
    para_oov = 0
    for token in vocabulary_tokens:
      token_id = vocab.token_to_id(token)
      if token in glove_embeddings:
        vocabulary_embeddings[token_id][:glove_embedding_size] = glove_embeddings[token]
      else:
        glove_oov += 1

    print('Glove OOV:', glove_oov, 'Para OOV', para_oov, 'Total', len(vocab))

    return vocabulary_embeddings

  input_vocabulary_embeddings = create_word_embeddings(input_vocabulary)
  output_vocabulary_embeddings = create_word_embeddings(output_vocabulary)
  output_vocabulary_schema_embeddings = None
  if output_vocabulary_schema:
    output_vocabulary_schema_embeddings = create_word_embeddings(output_vocabulary_schema)
  del glove_embeddings
  return input_vocabulary_embeddings, output_vocabulary_embeddings, output_vocabulary_schema_embeddings, input_embedding_size

class ATISModel(torch.nn.Module):
    """ Sequence-to-sequence model for predicting a SQL query given an utterance
        and an interaction prefix.
    """

    def __init__(
            self,
            params,
            input_vocabulary,
            output_vocabulary,
            output_vocabulary_schema,
            anonymizer):
        super().__init__()

        self.params = params

        if params.use_bert:
            self.model_bert, self.tokenizer, self.bert_config = utils_bert.get_bert(params)

        if 'atis' not in params.data_directory:
            if params.use_bert:
                input_vocabulary_embeddings, output_vocabulary_embeddings, output_vocabulary_schema_embeddings, input_embedding_size = load_word_embeddings(input_vocabulary, output_vocabulary, output_vocabulary_schema, params)

                # Create the output embeddings
                self.output_embedder = Embedder(params.output_embedding_size,
                                                name="output-embedding",
                                                initializer=output_vocabulary_embeddings,
                                                vocabulary=output_vocabulary,
                                                anonymizer=anonymizer,
                                                freeze=False)
                self.column_name_token_embedder = None
            else:
                input_vocabulary_embeddings, output_vocabulary_embeddings, output_vocabulary_schema_embeddings, input_embedding_size = load_word_embeddings(input_vocabulary, output_vocabulary, output_vocabulary_schema, params)

                params.input_embedding_size = input_embedding_size
                self.params.input_embedding_size = input_embedding_size

                # Create the input embeddings
                self.input_embedder = Embedder(params.input_embedding_size,
                                               name="input-embedding",
                                               initializer=input_vocabulary_embeddings,
                                               vocabulary=input_vocabulary,
                                               anonymizer=anonymizer,
                                               freeze=params.freeze)

                # Create the output embeddings
                self.output_embedder = Embedder(params.output_embedding_size,
                                                name="output-embedding",
                                                initializer=output_vocabulary_embeddings,
                                                vocabulary=output_vocabulary,
                                                anonymizer=anonymizer,
                                                freeze=False)

                self.column_name_token_embedder = Embedder(params.input_embedding_size,
                                                name="schema-embedding",
                                                initializer=output_vocabulary_schema_embeddings,
                                                vocabulary=output_vocabulary_schema,
                                                anonymizer=anonymizer,
                                                freeze=params.freeze)
        else:
            # Create the input embeddings
            self.input_embedder = Embedder(params.input_embedding_size,
                                           name="input-embedding",
                                           vocabulary=input_vocabulary,
                                           anonymizer=anonymizer,
                                           freeze=False)

            # Create the output embeddings
            self.output_embedder = Embedder(params.output_embedding_size,
                                            name="output-embedding",
                                            vocabulary=output_vocabulary,
                                            anonymizer=anonymizer,
                                            freeze=False)

            self.column_name_token_embedder = None

        # Create the encoder
        encoder_input_size = params.input_embedding_size
        encoder_output_size = params.encoder_state_size
        if params.use_bert:
            encoder_input_size = self.bert_config.hidden_size

        if params.discourse_level_lstm:
            encoder_input_size += params.encoder_state_size / 2

        self.utterance_encoder = Encoder(params.encoder_num_layers, encoder_input_size, encoder_output_size)

        # Positional embedder for utterances
        attention_key_size = params.encoder_state_size
        self.schema_attention_key_size = attention_key_size
        if params.state_positional_embeddings:
            attention_key_size += params.positional_embedding_size
            self.positional_embedder = Embedder(
                params.positional_embedding_size,
                name="positional-embedding",
                num_tokens=params.maximum_utterances)

        self.utterance_attention_key_size = attention_key_size


        # Create the discourse-level LSTM parameters
        if params.discourse_level_lstm:
            self.discourse_lstms = torch_utils.create_multilayer_lstm_params(1, params.encoder_state_size, params.encoder_state_size / 2, "LSTM-t")
            self.initial_discourse_state = torch_utils.add_params(tuple([params.encoder_state_size / 2]), "V-turn-state-0")

        # Snippet encoder
        final_snippet_size = 0
        if params.use_snippets and not params.previous_decoder_snippet_encoding:
            snippet_encoding_size = int(params.encoder_state_size / 2)
            final_snippet_size = params.encoder_state_size
            if params.snippet_age_embedding:
                snippet_encoding_size -= int(
                    params.snippet_age_embedding_size / 4)
                self.snippet_age_embedder = Embedder(
                    params.snippet_age_embedding_size,
                    name="snippet-age-embedding",
                    num_tokens=params.max_snippet_age_embedding)
                final_snippet_size = params.encoder_state_size + params.snippet_age_embedding_size / 2


            self.snippet_encoder = Encoder(params.snippet_num_layers,
                                           params.output_embedding_size,
                                           snippet_encoding_size)

        # Previous query Encoder
        if params.use_previous_query:
            self.query_encoder = Encoder(params.encoder_num_layers, params.output_embedding_size, params.encoder_state_size)

        self.final_snippet_size = final_snippet_size
        self.dropout = 0.

    def _encode_snippets(self, previous_query, snippets, input_schema):
        """ Computes a single vector representation for each snippet.

        Inputs:
            previous_query (list of str): Previous query in the interaction.
            snippets (list of Snippet): Snippets extracted from the previous

        Returns:
            list of Snippets, where the embedding is set to a vector.
        """
        startpoints = [snippet.startpos for snippet in snippets]
        endpoints = [snippet.endpos for snippet in snippets]
        assert len(startpoints) == 0 or min(startpoints) >= 0
        if input_schema:
            assert len(endpoints) == 0 or max(endpoints) <= len(previous_query)
        else:
            assert len(endpoints) == 0 or max(endpoints) < len(previous_query)

        snippet_embedder = lambda query_token: self.get_query_token_embedding(query_token, input_schema)
        if previous_query and snippets:
            _, previous_outputs = self.snippet_encoder(
                previous_query, snippet_embedder, dropout_amount=self.dropout)
            assert len(previous_outputs) == len(previous_query)

            for snippet in snippets:
                if input_schema:
                    embedding = torch.cat([previous_outputs[snippet.startpos],previous_outputs[snippet.endpos-1]], dim=0)
                else:
                    embedding = torch.cat([previous_outputs[snippet.startpos],previous_outputs[snippet.endpos]], dim=0)
                if self.params.snippet_age_embedding:
                    embedding = torch.cat([embedding, self.snippet_age_embedder(min(snippet.age, self.params.max_snippet_age_embedding - 1))], dim=0)
                snippet.set_embedding(embedding)

        return snippets

    def _initialize_discourse_states(self):
        discourse_state = self.initial_discourse_state

        discourse_lstm_states = []
        for lstm in self.discourse_lstms:
            hidden_size = lstm.weight_hh.size()[1]
            if lstm.weight_hh.is_cuda:
                h_0 = torch.cuda.FloatTensor(1,hidden_size).fill_(0)
                c_0 = torch.cuda.FloatTensor(1,hidden_size).fill_(0)
            else:
                h_0 = torch.zeros(1,hidden_size)
                c_0 = torch.zeros(1,hidden_size)
            discourse_lstm_states.append((h_0, c_0))

        return discourse_state, discourse_lstm_states

    def _add_positional_embeddings(self, hidden_states, utterances, group=False):
        grouped_states = []

        start_index = 0
        for utterance in utterances:
            grouped_states.append(hidden_states[start_index:start_index + len(utterance)])
            start_index += len(utterance)
        assert len(hidden_states) == sum([len(seq) for seq in grouped_states]) == sum([len(utterance) for utterance in utterances])

        new_states = []
        flat_sequence = []

        num_utterances_to_keep = min(self.params.maximum_utterances, len(utterances))
        for i, (states, utterance) in enumerate(zip(
                grouped_states[-num_utterances_to_keep:], utterances[-num_utterances_to_keep:])):
            positional_sequence = []
            index = num_utterances_to_keep - i - 1

            for state in states:
                positional_sequence.append(torch.cat([state, self.positional_embedder(index)], dim=0))

            assert len(positional_sequence) == len(utterance), \
                "Expected utterance and state sequence length to be the same, " \
                + "but they were " + str(len(utterance)) \
                + " and " + str(len(positional_sequence))

            if group:
                new_states.append(positional_sequence)
            else:
                new_states.extend(positional_sequence)
            flat_sequence.extend(utterance)

        return new_states, flat_sequence

    def build_optim(self):
        params_trainer = []
        params_bert_trainer = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                if 'model_bert' in name:
                    params_bert_trainer.append(param)
                else:
                    params_trainer.append(param)
        self.trainer = torch.optim.Adam(params_trainer, lr=self.params.initial_learning_rate)
        if self.params.fine_tune_bert:
            self.bert_trainer = torch.optim.Adam(params_bert_trainer, lr=self.params.lr_bert)

    def set_dropout(self, value):
        """ Sets the dropout to a specified value.

        Inputs:
            value (float): Value to set dropout to.
        """
        self.dropout = value

    def set_learning_rate(self, value):
        """ Sets the learning rate for the trainer.

        Inputs:
            value (float): The new learning rate.
        """
        for param_group in self.trainer.param_groups:
            param_group['lr'] = value

    def save(self, filename):
        """ Saves the model to the specified filename.

        Inputs:
            filename (str): The filename to save to.
        """
        torch.save(self.state_dict(), filename)

    def load(self, filename):
        """ Loads saved parameters into the parameter collection.

        Inputs:
            filename (str): Name of file containing parameters.
        """
        self.load_state_dict(torch.load(filename))
        print("Loaded model from file " + filename)


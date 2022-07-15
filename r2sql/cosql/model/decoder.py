""" Decoder for the SQL generation problem."""

from collections import namedtuple
import copy
import json
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from . import torch_utils

from .token_predictor import PredictionInput, PredictionInputWithSchema, PredictionStepInputWithSchema
from .beam_search import BeamSearch
import data_util.snippets as snippet_handler
from . import embedder
from data_util.vocabulary import EOS_TOK, UNK_TOK
import math

def flatten_distribution(distribution_map, probabilities):
    """ Flattens a probability distribution given a map of "unique" values.
        All values in distribution_map with the same value should get the sum
        of the probabilities.

        Arguments:
            distribution_map (list of str): List of values to get the probability for.
            probabilities (np.ndarray): Probabilities corresponding to the values in
                distribution_map.

        Returns:
            list, np.ndarray of the same size where probabilities for duplicates
                in distribution_map are given the sum of the probabilities in probabilities.
    """
    assert len(distribution_map) == len(probabilities)
    if len(distribution_map) != len(set(distribution_map)):
        idx_first_dup = 0
        seen_set = set()
        for i, tok in enumerate(distribution_map):
            if tok in seen_set:
                idx_first_dup = i
                break
            seen_set.add(tok)
        new_dist_map = distribution_map[:idx_first_dup] + list(
            set(distribution_map) - set(distribution_map[:idx_first_dup]))
        assert len(new_dist_map) == len(set(new_dist_map))
        new_probs = np.array(
            probabilities[:idx_first_dup] \
            + [0. for _ in range(len(set(distribution_map)) \
                                 - idx_first_dup)])
        assert len(new_probs) == len(new_dist_map)

        for i, token_name in enumerate(
                distribution_map[idx_first_dup:]):
            if token_name not in new_dist_map:
                new_dist_map.append(token_name)

            new_index = new_dist_map.index(token_name)
            new_probs[new_index] += probabilities[i +
                                                  idx_first_dup]
        new_probs = new_probs.tolist()
    else:
        new_dist_map = distribution_map
        new_probs = probabilities

    assert len(new_dist_map) == len(new_probs)

    return new_dist_map, new_probs

class SQLPrediction(namedtuple('SQLPrediction',
                               ('predictions',
                                'sequence',
                                'probability',
                                'beam'))):
    """Contains prediction for a sequence."""
    __slots__ = ()

    def __str__(self):
        return str(self.probability) + "\t" + " ".join(self.sequence)

Count = 0

class SequencePredictorWithSchema(torch.nn.Module):
    """ Predicts a sequence.

    Attributes:
        lstms (list of dy.RNNBuilder): The RNN used.
        token_predictor (TokenPredictor): Used to actually predict tokens.
    """
    def __init__(self,
                 params,
                 input_size,
                 output_embedder,
                 column_name_token_embedder,
                 token_predictor):
        super().__init__()

        self.lstms = torch_utils.create_multilayer_lstm_params(params.decoder_num_layers, input_size, params.decoder_state_size, "LSTM-d")
        self.token_predictor = token_predictor
        self.output_embedder = output_embedder
        self.column_name_token_embedder = column_name_token_embedder
        self.start_token_embedding = torch_utils.add_params((params.output_embedding_size,), "y-0")

        self.input_size = input_size
        self.params = params

        self.params.use_turn_num = False

        if self.params.use_turn_num:
            assert params.decoder_num_layers <= 2
            state_size = params.decoder_state_size
            self.layer1_c_0 = nn.Linear(state_size+5, state_size)
            self.layer1_h_0 = nn.Linear(state_size+5, state_size)
            if params.decoder_num_layers == 2:
                self.layer2_c_0 = nn.Linear(state_size+5, state_size)
                self.layer2_h_0 = nn.Linear(state_size+5, state_size)
        

    def _initialize_decoder_lstm(self, encoder_state, utterance_index):
        decoder_lstm_states = []
        #print('utterance_index', utterance_index)
        #print('self.params.decoder_num_layers', self.params.decoder_num_layers)
        if self.params.use_turn_num:
            if utterance_index >= 4:
                utterance_index = 4;
            turn_number_embedding = torch.zeros([1, 5]).cuda()
            turn_number_embedding[0, utterance_index] = 1
        #print('turn_number_embedding', turn_number_embedding)

        for i, lstm in enumerate(self.lstms):
            encoder_layer_num = 0
            if len(encoder_state[0]) > 1:
                encoder_layer_num = i

            # check which one is h_0, which is c_0
            c_0 = encoder_state[0][encoder_layer_num].view(1,-1)
            h_0 = encoder_state[1][encoder_layer_num].view(1,-1)

            if self.params.use_turn_num:
                if i == 0:
                    c_0 = self.layer1_c_0( torch.cat([c_0, turn_number_embedding], -1) )
                    h_0 = self.layer1_h_0( torch.cat([h_0, turn_number_embedding], -1) )
                else:
                    c_0 = self.layer2_c_0( torch.cat([c_0, turn_number_embedding], -1) )
                    h_0 = self.layer2_h_0( torch.cat([h_0, turn_number_embedding], -1) )

            decoder_lstm_states.append((h_0, c_0))
        return decoder_lstm_states

    def get_output_token_embedding(self, output_token, input_schema, snippets):
        if self.params.use_snippets and snippet_handler.is_snippet(output_token):
            output_token_embedding = embedder.bow_snippets(output_token, snippets, self.output_embedder, input_schema)
        else:
            if input_schema:
                assert self.output_embedder.in_vocabulary(output_token) or input_schema.in_vocabulary(output_token, surface_form=True)
                if self.output_embedder.in_vocabulary(output_token):
                    output_token_embedding = self.output_embedder(output_token)
                else:
                    output_token_embedding = input_schema.column_name_embedder(output_token, surface_form=True)
            else:
                output_token_embedding = self.output_embedder(output_token)
        return output_token_embedding

    def get_decoder_input(self, output_token_embedding, prediction):
        if self.params.use_schema_attention and self.params.use_query_attention:
            decoder_input = torch.cat([output_token_embedding, prediction.utterance_attention_results.vector, prediction.schema_attention_results.vector, prediction.query_attention_results.vector], dim=0)
        elif self.params.use_schema_attention:
            decoder_input = torch.cat([output_token_embedding, prediction.utterance_attention_results.vector, prediction.schema_attention_results.vector], dim=0)
        else:
            decoder_input = torch.cat([output_token_embedding, prediction.utterance_attention_results.vector], dim=0)
        return decoder_input

    def forward(self,
                final_encoder_state,
                encoder_states,
                schema_states,
                max_generation_length,
                utterance_index=None,
                snippets=None,
                gold_sequence=None,
                input_sequence=None,
                previous_queries=None,
                previous_query_states=None,
                input_schema=None,
                dropout_amount=0.):
        global Count
        """ Generates a sequence. """
        index = 0

        context_vector_size = self.input_size - self.params.output_embedding_size

        # Decoder states: just the initialized decoder.
        # Current input to decoder: phi(start_token) ; zeros the size of the
        # context vector
        predictions = []
        sequence = []
        probability = 1.

        decoder_states = self._initialize_decoder_lstm(final_encoder_state, utterance_index)

        if self.start_token_embedding.is_cuda:
            decoder_input = torch.cat([self.start_token_embedding, torch.cuda.FloatTensor(context_vector_size).fill_(0)], dim=0)
        else:
            decoder_input = torch.cat([self.start_token_embedding, torch.zeros(context_vector_size)], dim=0)

        beam_search = BeamSearch(is_end_of_sequence=self.is_end_of_sequence, max_steps=max_generation_length, beam_size=10)
        prediction_step_input = PredictionStepInputWithSchema(
            index=index,
            decoder_input=decoder_input,
            decoder_states=decoder_states,
            encoder_states=encoder_states,
            schema_states=schema_states,
            snippets=snippets,
            gold_sequence=gold_sequence,
            input_sequence=input_sequence,
            previous_queries=previous_queries,
            previous_query_states=previous_query_states,
            input_schema=input_schema,
            dropout_amount=dropout_amount,
            predictions=predictions,
        )

        sequence, probability, final_state, beam = beam_search.search(start_state=prediction_step_input, step_function=self.beam_search_step_function, append_token_function=self.beam_search_append_token)

        sum_p = 0
        for x in beam:
        #    print('math.exp(x[0])', math.exp(x[0]))
            #print('x[1]', x[1].sequence)
            sum_p += math.exp(x[0])
        #print('sum_p', sum_p)
        assert sum_p <= 1.2

        '''if len(sequence) <= 2:
            print('beam', beam)
            for x in beam:
                print('x[0]', x[0])
                print('x[1]', x[1].sequence)
            print('='*20)
            Count+=1
            assert Count <= 10'''

        return SQLPrediction(final_state.predictions,
                             sequence,
                             probability,
                             beam)

    def beam_search_step_function(self, prediction_step_input):
        # get decoded token probabilities
        prediction, tokens, token_probabilities, decoder_states = self.decode_next_token(prediction_step_input)
        return tokens, token_probabilities, (prediction, decoder_states)

    def beam_search_append_token(self, prediction_step_input, step_function_output, token_to_append, token_log_probability):
        output_token_embedding = self.get_output_token_embedding(token_to_append, prediction_step_input.input_schema, prediction_step_input.snippets)
        decoder_input = self.get_decoder_input(output_token_embedding, step_function_output[0])

        predictions = prediction_step_input.predictions.copy()
        predictions.append(step_function_output[0])
        new_state = prediction_step_input._replace(
            index=prediction_step_input.index+1,
            decoder_input=decoder_input,
            predictions=predictions,
            decoder_states=step_function_output[1],
        )

        return new_state

    def is_end_of_sequence(self, prediction_step_input, max_generation_length, sequence):
        if prediction_step_input.gold_sequence:
            return prediction_step_input.index >= len(prediction_step_input.gold_sequence)
        else:
            return prediction_step_input.index >= max_generation_length or (len(sequence) > 0 and sequence[-1] == EOS_TOK)

    def decode_next_token(self, prediction_step_input):
        (index,
         decoder_input,
         decoder_states,
         encoder_states,
         schema_states,
         snippets,
         gold_sequence,
         input_sequence,
         previous_queries,
         previous_query_states,
         input_schema,
         dropout_amount,
         predictions) = prediction_step_input

        _, decoder_state, decoder_states = torch_utils.forward_one_multilayer(self.lstms, decoder_input, decoder_states, dropout_amount)
        prediction_input = PredictionInputWithSchema(decoder_state=decoder_state,
                                                     input_hidden_states=encoder_states,
                                                     schema_states=schema_states,
                                                     snippets=snippets,
                                                     input_sequence=input_sequence,
                                                     previous_queries=previous_queries,
                                                     previous_query_states=previous_query_states,
                                                     input_schema=input_schema)

        prediction = self.token_predictor(prediction_input, dropout_amount=dropout_amount)

        #gold_sequence = None

        if gold_sequence:
            decoded_token = gold_sequence[index]
            distribution_map = [decoded_token]
            probabilities = [1.0]

        else:#/mnt/lichaochao/rqy/editsql_data/glove.840B.300d.txt
            assert prediction.scores.dim() == 1
            probabilities = F.softmax(prediction.scores, dim=0).cpu().data.numpy().tolist()

            distribution_map = prediction.aligned_tokens
            assert len(probabilities) == len(distribution_map)

            if self.params.use_previous_query and self.params.use_copy_switch and len(previous_queries) > 0:
                assert prediction.query_scores.dim() == 1
                query_token_probabilities = F.softmax(prediction.query_scores, dim=0).cpu().data.numpy().tolist()

                query_token_distribution_map = prediction.query_tokens

                assert len(query_token_probabilities) == len(query_token_distribution_map)

                copy_switch = prediction.copy_switch.cpu().data.numpy()

                # Merge the two
                probabilities = ((np.array(probabilities) * (1 - copy_switch)).tolist() + 
                                 (np.array(query_token_probabilities) * copy_switch).tolist()
                                 )
                distribution_map =  distribution_map + query_token_distribution_map
                assert len(probabilities) == len(distribution_map)

            # Get a new probabilities and distribution_map consolidating duplicates
            distribution_map, probabilities = flatten_distribution(distribution_map, probabilities) 

            # Modify the probability distribution so that the UNK token can never be produced
            probabilities[distribution_map.index(UNK_TOK)] = 0.

        return prediction, distribution_map, probabilities, decoder_states

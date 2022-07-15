"""Predicts a token."""

from collections import namedtuple

import torch
import torch.nn.functional as F
from . import torch_utils

from .attention import Attention, AttentionResult

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PredictionInput(namedtuple('PredictionInput',
                                 ('decoder_state',
                                  'input_hidden_states',
                                  'snippets',
                                  'input_sequence'))):
    """ Inputs to the token predictor. """
    __slots__ = ()

class PredictionStepInputWithSchema(namedtuple('PredictionStepInputWithSchema',
                                 ('index',
                                  'decoder_input',
                                  'decoder_states',
                                  'encoder_states',
                                  'schema_states',
                                  'snippets',
                                  'gold_sequence',
                                  'input_sequence',
                                  'previous_queries',
                                  'previous_query_states',
                                  'input_schema',
                                  'dropout_amount',
                                  'predictions'))):
    """ Inputs to the next token predictor. """
    __slots__ = ()

class PredictionInputWithSchema(namedtuple('PredictionInputWithSchema',
                                 ('decoder_state',
                                  'input_hidden_states',
                                  'schema_states',
                                  'snippets',
                                  'input_sequence',
                                  'previous_queries',
                                  'previous_query_states',
                                  'input_schema'))):
    """ Inputs to the token predictor. """
    __slots__ = ()


class TokenPrediction(namedtuple('TokenPrediction',
                                 ('scores',
                                  'aligned_tokens',
                                  'utterance_attention_results',
                                  'schema_attention_results',
                                  'query_attention_results',
                                  'copy_switch',
                                  'query_scores',
                                  'query_tokens',
                                  'decoder_state'))):

    """A token prediction."""
    __slots__ = ()

def score_snippets(snippets, scorer):
    """ Scores snippets given a scorer.

    Inputs:
        snippets (list of Snippet): The snippets to score.
        scorer (dy.Expression): Dynet vector against which to score  the snippets.

    Returns:
        dy.Expression, list of str, where the first is the scores and the second
            is the names of the snippets that were scored.
    """
    snippet_expressions = [snippet.embedding for snippet in snippets]
    all_snippet_embeddings = torch.stack(snippet_expressions, dim=1)

    scores = torch.t(torch.mm(torch.t(scorer), all_snippet_embeddings))

    if scores.size()[0] != len(snippets):
        raise ValueError("Got " + str(scores.size()[0]) + " scores for " + str(len(snippets)) + " snippets")

    return scores, [snippet.name for snippet in snippets]

def score_schema_tokens(input_schema, schema_states, scorer):
    # schema_states: emd_dim x num_tokens
    scores = torch.t(torch.mm(torch.t(scorer), schema_states))   # num_tokens x 1
    if scores.size()[0] != len(input_schema):
        raise ValueError("Got " + str(scores.size()[0]) + " scores for " + str(len(input_schema)) + " schema tokens")
    return scores, input_schema.column_names_surface_form

def score_query_tokens(previous_query, previous_query_states, scorer):
    scores = torch.t(torch.mm(torch.t(scorer), previous_query_states))   # num_tokens x 1
    if scores.size()[0] != len(previous_query):
        raise ValueError("Got " + str(scores.size()[0]) + " scores for " + str(len(previous_query)) + " query tokens")
    return scores, previous_query

class TokenPredictor(torch.nn.Module):
    """ Predicts a token given a (decoder) state.

    Attributes:
        vocabulary (Vocabulary): A vocabulary object for the output.
        attention_module (Attention): An attention module.
        state_transformation_weights (dy.Parameters): Transforms the input state
            before predicting a token.
        vocabulary_weights (dy.Parameters): Final layer weights.
        vocabulary_biases (dy.Parameters): Final layer biases.
    """

    def __init__(self, params, vocabulary, attention_key_size):
        super().__init__()
        self.params = params
        self.vocabulary = vocabulary
        self.attention_module = Attention(params.decoder_state_size, attention_key_size, attention_key_size)
        self.state_transform_weights = torch_utils.add_params((params.decoder_state_size + attention_key_size, params.decoder_state_size), "weights-state-transform")
        self.vocabulary_weights = torch_utils.add_params((params.decoder_state_size, len(vocabulary)), "weights-vocabulary")
        self.vocabulary_biases = torch_utils.add_params(tuple([len(vocabulary)]), "biases-vocabulary")

    def _get_intermediate_state(self, state, dropout_amount=0.):
        intermediate_state = torch.tanh(torch_utils.linear_layer(state, self.state_transform_weights))
        return F.dropout(intermediate_state, dropout_amount)

    def _score_vocabulary_tokens(self, state):
        scores = torch.t(torch_utils.linear_layer(state, self.vocabulary_weights, self.vocabulary_biases))

        if scores.size()[0] != len(self.vocabulary.inorder_tokens):
            raise ValueError("Got " + str(scores.size()[0]) + " scores for " + str(len(self.vocabulary.inorder_tokens)) + " vocabulary items")

        return scores, self.vocabulary.inorder_tokens

    def forward(self, prediction_input, dropout_amount=0.):
        decoder_state = prediction_input.decoder_state
        input_hidden_states = prediction_input.input_hidden_states

        attention_results = self.attention_module(decoder_state, input_hidden_states)

        state_and_attn = torch.cat([decoder_state, attention_results.vector], dim=0)

        intermediate_state = self._get_intermediate_state(state_and_attn, dropout_amount=dropout_amount)
        vocab_scores, vocab_tokens = self._score_vocabulary_tokens(intermediate_state)

        return TokenPrediction(vocab_scores, vocab_tokens, attention_results, decoder_state)


class SchemaTokenPredictor(TokenPredictor):
    """ Token predictor that also predicts snippets.

    Attributes:
        snippet_weights (dy.Parameter): Weights for scoring snippets against some
            state.
    """

    def __init__(self, params, vocabulary, utterance_attention_key_size, schema_attention_key_size, snippet_size):
        TokenPredictor.__init__(self, params, vocabulary, utterance_attention_key_size)
        if params.use_snippets:
            if snippet_size <= 0:
                raise ValueError("Snippet size must be greater than zero; was " + str(snippet_size))
            self.snippet_weights = torch_utils.add_params((params.decoder_state_size, snippet_size), "weights-snippet")

        if params.use_schema_attention:
            self.utterance_attention_module = self.attention_module
            self.schema_attention_module = Attention(params.decoder_state_size, schema_attention_key_size, schema_attention_key_size)

        if self.params.use_query_attention:
            self.query_attention_module = Attention(params.decoder_state_size, params.encoder_state_size, params.encoder_state_size)
            self.start_query_attention_vector = torch_utils.add_params((params.encoder_state_size,), "start_query_attention_vector")

        if params.use_schema_attention and self.params.use_query_attention:
            self.state_transform_weights = torch_utils.add_params((params.decoder_state_size + utterance_attention_key_size + schema_attention_key_size + params.encoder_state_size, params.decoder_state_size), "weights-state-transform")
        elif params.use_schema_attention:
            self.state_transform_weights = torch_utils.add_params((params.decoder_state_size + utterance_attention_key_size + schema_attention_key_size, params.decoder_state_size), "weights-state-transform")

        # Use lstm schema encoder
        self.schema_token_weights = torch_utils.add_params((params.decoder_state_size, schema_attention_key_size), "weights-schema-token")

        if self.params.use_previous_query:
            self.query_token_weights = torch_utils.add_params((params.decoder_state_size, self.params.encoder_state_size), "weights-query-token")

        if self.params.use_copy_switch:
            if self.params.use_query_attention:
                self.state2copyswitch_transform_weights = torch_utils.add_params((params.decoder_state_size + utterance_attention_key_size + schema_attention_key_size + params.encoder_state_size, 1), "weights-state-transform")
            else:
                self.state2copyswitch_transform_weights = torch_utils.add_params((params.decoder_state_size + utterance_attention_key_size + schema_attention_key_size, 1), "weights-state-transform")

    def _get_snippet_scorer(self, state):
        scorer = torch.t(torch_utils.linear_layer(state, self.snippet_weights))
        return scorer

    def _get_schema_token_scorer(self, state):
        scorer = torch.t(torch_utils.linear_layer(state, self.schema_token_weights))
        return scorer

    def _get_query_token_scorer(self, state):
        scorer = torch.t(torch_utils.linear_layer(state, self.query_token_weights))
        return scorer

    def _get_copy_switch(self, state):
        copy_switch = torch.sigmoid(torch_utils.linear_layer(state, self.state2copyswitch_transform_weights))
        return copy_switch.squeeze()

    def forward(self, prediction_input, dropout_amount=0.):
        decoder_state = prediction_input.decoder_state
        input_hidden_states = prediction_input.input_hidden_states
        snippets = prediction_input.snippets

        input_schema = prediction_input.input_schema
        schema_states = prediction_input.schema_states

        if self.params.use_schema_attention:
            schema_attention_results = self.schema_attention_module(decoder_state, schema_states)
            utterance_attention_results = self.utterance_attention_module(decoder_state, input_hidden_states)
        else:
            utterance_attention_results = self.attention_module(decoder_state, input_hidden_states)
            schema_attention_results = None

        query_attention_results = None
        if self.params.use_query_attention:
            previous_query_states = prediction_input.previous_query_states
            if len(previous_query_states) > 0:
                query_attention_results = self.query_attention_module(decoder_state, previous_query_states[-1])
            else:
                query_attention_results = self.start_query_attention_vector
                query_attention_results = AttentionResult(None, None, query_attention_results)

        if self.params.use_schema_attention and self.params.use_query_attention:
            state_and_attn = torch.cat([decoder_state, utterance_attention_results.vector, schema_attention_results.vector, query_attention_results.vector], dim=0)
        elif self.params.use_schema_attention:
            state_and_attn = torch.cat([decoder_state, utterance_attention_results.vector, schema_attention_results.vector], dim=0)
        else:
            state_and_attn = torch.cat([decoder_state, utterance_attention_results.vector], dim=0)

        intermediate_state = self._get_intermediate_state(state_and_attn, dropout_amount=dropout_amount)
        vocab_scores, vocab_tokens = self._score_vocabulary_tokens(intermediate_state)

        final_scores = vocab_scores
        aligned_tokens = []
        aligned_tokens.extend(vocab_tokens)

        if self.params.use_snippets and snippets:
            snippet_scores, snippet_tokens = score_snippets(snippets, self._get_snippet_scorer(intermediate_state))
            final_scores = torch.cat([final_scores, snippet_scores], dim=0)
            aligned_tokens.extend(snippet_tokens)

        schema_states = torch.stack(schema_states, dim=1)
        schema_scores, schema_tokens = score_schema_tokens(input_schema, schema_states, self._get_schema_token_scorer(intermediate_state))

        final_scores = torch.cat([final_scores, schema_scores], dim=0)
        aligned_tokens.extend(schema_tokens)

        # Previous Queries
        previous_queries = prediction_input.previous_queries
        previous_query_states = prediction_input.previous_query_states

        copy_switch = None
        query_scores = None
        query_tokens = None
        if self.params.use_previous_query and len(previous_queries) > 0:
            if self.params.use_copy_switch:
                copy_switch = self._get_copy_switch(state_and_attn)
            for turn, (previous_query, previous_query_state) in enumerate(zip(previous_queries, previous_query_states)):
                assert len(previous_query) == len(previous_query_state)
                previous_query_state = torch.stack(previous_query_state, dim=1)
                query_scores, query_tokens = score_query_tokens(previous_query, previous_query_state, self._get_query_token_scorer(intermediate_state))
                query_scores = query_scores.squeeze()

        final_scores = final_scores.squeeze()

        return TokenPrediction(final_scores, aligned_tokens, utterance_attention_results, schema_attention_results, query_attention_results, copy_switch, query_scores, query_tokens, decoder_state)


class SnippetTokenPredictor(TokenPredictor):
    """ Token predictor that also predicts snippets.

    Attributes:
        snippet_weights (dy.Parameter): Weights for scoring snippets against some
            state.
    """

    def __init__(self, params, vocabulary, attention_key_size, snippet_size):
        TokenPredictor.__init__(self, params, vocabulary, attention_key_size)
        if snippet_size <= 0:
            raise ValueError("Snippet size must be greater than zero; was " + str(snippet_size))

        self.snippet_weights = torch_utils.add_params((params.decoder_state_size, snippet_size), "weights-snippet")

    def _get_snippet_scorer(self, state):
        scorer = torch.t(torch_utils.linear_layer(state, self.snippet_weights))

        return scorer

    def forward(self, prediction_input, dropout_amount=0.):
        decoder_state = prediction_input.decoder_state
        input_hidden_states = prediction_input.input_hidden_states
        snippets = prediction_input.snippets

        attention_results = self.attention_module(decoder_state,
                                                  input_hidden_states)

        state_and_attn = torch.cat([decoder_state, attention_results.vector], dim=0)


        intermediate_state = self._get_intermediate_state(
            state_and_attn, dropout_amount=dropout_amount)
        vocab_scores, vocab_tokens = self._score_vocabulary_tokens(
            intermediate_state)

        final_scores = vocab_scores
        aligned_tokens = []
        aligned_tokens.extend(vocab_tokens)

        if snippets:
            snippet_scores, snippet_tokens = score_snippets(
                snippets,
                self._get_snippet_scorer(intermediate_state))

            final_scores = torch.cat([final_scores, snippet_scores], dim=0)
            aligned_tokens.extend(snippet_tokens)

        final_scores = final_scores.squeeze()

        return TokenPrediction(final_scores, aligned_tokens, attention_results, None, None, None, None, None, decoder_state)


class AnonymizationTokenPredictor(TokenPredictor):
    """ Token predictor that also predicts anonymization tokens.

    Attributes:
        anonymizer (Anonymizer): The anonymization object.

    """

    def __init__(self, params, vocabulary, attention_key_size, anonymizer):
        TokenPredictor.__init__(self, params, vocabulary, attention_key_size)
        if not anonymizer:
            raise ValueError("Expected an anonymizer, but was None")
        self.anonymizer = anonymizer

    def _score_anonymized_tokens(self,
                                 input_sequence,
                                 attention_scores):
        scores = []
        tokens = []
        for i, token in enumerate(input_sequence):
            if self.anonymizer.is_anon_tok(token):
                scores.append(attention_scores[i])
                tokens.append(token)

        if len(scores) > 0:
            if len(scores) != len(tokens):
                raise ValueError("Got " + str(len(scores)) + " scores for "
                                 + str(len(tokens)) + " anonymized tokens")

            anonymized_scores = torch.cat(scores, dim=0)
            if anonymized_scores.dim() == 1:
                anonymized_scores = anonymized_scores.unsqueeze(1)
            return anonymized_scores, tokens
        else:
            return None, []

    def forward(self,
                 prediction_input,
                 dropout_amount=0.):
        decoder_state = prediction_input.decoder_state
        input_hidden_states = prediction_input.input_hidden_states
        input_sequence = prediction_input.input_sequence
        assert input_sequence

        attention_results = self.attention_module(decoder_state,
                                                  input_hidden_states)

        state_and_attn = torch.cat([decoder_state, attention_results.vector], dim=0)

        intermediate_state = self._get_intermediate_state(
            state_and_attn, dropout_amount=dropout_amount)
        vocab_scores, vocab_tokens = self._score_vocabulary_tokens(
            intermediate_state)

        final_scores = vocab_scores
        aligned_tokens = []
        aligned_tokens.extend(vocab_tokens)

        anonymized_scores, anonymized_tokens = self._score_anonymized_tokens(
            input_sequence,
            attention_results.scores)

        if anonymized_scores:
            final_scores = torch.cat([final_scores, anonymized_scores], dim=0)
            aligned_tokens.extend(anonymized_tokens)

        final_scores = final_scores.squeeze()

        return TokenPrediction(final_scores, aligned_tokens, attention_results, None, None, None, None, None, decoder_state)


# For Atis
class SnippetAnonymizationTokenPredictor(SnippetTokenPredictor, AnonymizationTokenPredictor):
    """ Token predictor that both anonymizes and scores snippets."""

    def __init__(self, params, vocabulary, attention_key_size, snippet_size, anonymizer):
        AnonymizationTokenPredictor.__init__(self, params, vocabulary, attention_key_size, anonymizer)
        SnippetTokenPredictor.__init__(self, params, vocabulary, attention_key_size, snippet_size)

    def forward(self, prediction_input, dropout_amount=0.):
        decoder_state = prediction_input.decoder_state
        assert prediction_input.input_sequence

        snippets = prediction_input.snippets

        input_hidden_states = prediction_input.input_hidden_states

        attention_results = self.attention_module(decoder_state,
                                                  prediction_input.input_hidden_states)

        state_and_attn = torch.cat([decoder_state, attention_results.vector], dim=0)

        intermediate_state = self._get_intermediate_state(state_and_attn, dropout_amount=dropout_amount)

        # Vocabulary tokens
        final_scores, vocab_tokens = self._score_vocabulary_tokens(intermediate_state)

        aligned_tokens = []
        aligned_tokens.extend(vocab_tokens)

        # Snippets
        if snippets:
            snippet_scores, snippet_tokens = score_snippets(
                snippets,
                self._get_snippet_scorer(intermediate_state))

            final_scores = torch.cat([final_scores, snippet_scores], dim=0)
            aligned_tokens.extend(snippet_tokens)

        # Anonymized tokens
        anonymized_scores, anonymized_tokens = self._score_anonymized_tokens(
            prediction_input.input_sequence,
            attention_results.scores)

        if anonymized_scores is not None:
            final_scores = torch.cat([final_scores, anonymized_scores], dim=0)
            aligned_tokens.extend(anonymized_tokens)

        final_scores = final_scores.squeeze()

        return TokenPrediction(final_scores, aligned_tokens, attention_results, None, None, None, None, None, decoder_state)

def construct_token_predictor(params,
                              vocabulary,
                              utterance_attention_key_size,
                              schema_attention_key_size,
                              snippet_size,
                              anonymizer=None):
    """ Constructs a token predictor given the parameters.

    Inputs:
        parameter_collection (dy.ParameterCollection): Contains the parameters.
        params (dictionary): Contains the command line parameters/hyperparameters.
        vocabulary (Vocabulary): Vocabulary object for output generation.
        attention_key_size (int): The size of the attention keys.
        anonymizer (Anonymizer): An anonymization object.
    """


    if not anonymizer and not params.previous_decoder_snippet_encoding:
        print('using SchemaTokenPredictor')
        return SchemaTokenPredictor(params, vocabulary, utterance_attention_key_size, schema_attention_key_size, snippet_size)
    elif params.use_snippets and anonymizer and not params.previous_decoder_snippet_encoding:
        print('using SnippetAnonymizationTokenPredictor')
        return SnippetAnonymizationTokenPredictor(params,
                                                  vocabulary,
                                                  utterance_attention_key_size,
                                                  snippet_size,
                                                  anonymizer)
    else:
        print('Unknown token_predictor')
        exit()

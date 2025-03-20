from transformers import PreTrainedTokenizer

import constants


def linearized_input_data(linearized_nodes, tokenizer: PreTrainedTokenizer, special_tokens,
                          metadatas=None,
                          prefix=None, lower=False):
    if special_tokens is not None and len(special_tokens):
        assert constants.METADATA_TOKEN in special_tokens and constants.NODE_TOKEN in special_tokens

    text = []
    if prefix is not None and len(prefix) > 0:
        if lower:
            prefix = prefix.lower()
        text.append(prefix.strip())

    if metadatas is not None and len(metadatas):
        for metadata in metadatas:
            text.append(constants.METADATA_TOKEN + " " + metadata)

    for node in linearized_nodes:
        text.append(constants.NODE_TOKEN + " " + node)

    text = " ".join(text)
    tokens = tokenizer.tokenize(text)

    return tokens


def leattice_linearized_input_data(linearized_nodes, tokenizer: PreTrainedTokenizer, special_tokens,
                                   metadatas=None,
                                   prefix=None, lower=False):
    if special_tokens is not None and len(special_tokens):
        assert constants.METADATA_TOKEN in special_tokens and constants.NODE_TOKEN in special_tokens

    text = []
    if prefix is not None:
        if lower:
            prefix = prefix.lower()
        text.append(prefix)

    if metadatas is not None and len(metadatas):
        for metadata in metadatas:
            text.append(constants.METADATA_TOKEN + " " + metadata)

    for node in linearized_nodes:
        text.append(constants.NODE_TOKEN + " " + node)

    text = " ".join(text)
    tokens = tokenizer.tokenize(text)

    return tokens

import torch
from collections import OrderedDict

BERT_VOCAB_SIZE = 30522


def get_match_value(name, state_dict_numpy, prefix):
    """
    Need be overridden towards different models, here for UnifiedTransformer Model
    """
    if name == 'embedder.token_embedding.weight':
        return state_dict_numpy[f'{prefix}embeddings.word_embeddings.weight']
    elif name == 'embedder.pos_embedding.weight':
        return state_dict_numpy[f'{prefix}embeddings.position_embeddings.weight']
    elif name == 'embedder.type_embedding.weight':
        return state_dict_numpy.get(f'{prefix}embeddings.token_type_embeddings.weight')
    elif name == 'embedder.turn_embedding.weight':
        return None
    elif name == 'embed_layer_norm.weight':
        if f'{prefix}embeddings.LayerNorm.weight' in state_dict_numpy:
            return state_dict_numpy[f'{prefix}embeddings.LayerNorm.weight']
        else:
            return state_dict_numpy[f'{prefix}embeddings.LayerNorm.gamma']
    elif name == 'embed_layer_norm.bias':
        if f'{prefix}embeddings.LayerNorm.bias' in state_dict_numpy:
            return state_dict_numpy[f'{prefix}embeddings.LayerNorm.bias']
        else:
            return state_dict_numpy[f'{prefix}embeddings.LayerNorm.beta']
    elif name == 'pooler.0.weight':
        return state_dict_numpy.get(f'{prefix}pooler.dense.weight')
    elif name == 'pooler.0.bias':
        return state_dict_numpy.get(f'{prefix}pooler.dense.bias')
    elif name == 'mlm_transform.0.weight':
        return state_dict_numpy.get('cls.predictions.transform.dense.weight')
    elif name == 'mlm_transform.0.bias':
        return state_dict_numpy.get('cls.predictions.transform.dense.bias')
    elif name == 'mlm_transform.2.weight':
        if 'cls.predictions.transform.LayerNorm.weight' in state_dict_numpy:
            return state_dict_numpy.get('cls.predictions.transform.LayerNorm.weight')
        else:
            return state_dict_numpy.get('cls.predictions.transform.LayerNorm.gamma')
    elif name == 'mlm_transform.2.bias':
        if 'cls.predictions.transform.LayerNorm.bias' in state_dict_numpy:
            return state_dict_numpy.get('cls.predictions.transform.LayerNorm.bias')
        else:
            return state_dict_numpy.get('cls.predictions.transform.LayerNorm.beta')
    elif name == 'mlm_bias':
        return state_dict_numpy.get('cls.predictions.bias')
    else:
        num = name.split('.')[1]
        assert num in [str(i) for i in range(12)]
        if name == f'layers.{num}.attn.linear_qkv.weight':
            q = state_dict_numpy[f'{prefix}encoder.layer.{num}.attention.self.query.weight']
            k = state_dict_numpy[f'{prefix}encoder.layer.{num}.attention.self.key.weight']
            v = state_dict_numpy[f'{prefix}encoder.layer.{num}.attention.self.value.weight']
            qkv_weight = torch.cat([q, k, v], dim=0)
            return qkv_weight
        elif name == f'layers.{num}.attn.linear_qkv.bias':
            q = state_dict_numpy[f'{prefix}encoder.layer.{num}.attention.self.query.bias']
            k = state_dict_numpy[f'{prefix}encoder.layer.{num}.attention.self.key.bias']
            v = state_dict_numpy[f'{prefix}encoder.layer.{num}.attention.self.value.bias']
            qkv_bias = torch.cat([q, k, v], dim=0)
            return qkv_bias
        elif name == f'layers.{num}.attn.linear_out.weight':
            return state_dict_numpy[f'{prefix}encoder.layer.{num}.attention.output.dense.weight']
        elif name == f'layers.{num}.attn.linear_out.bias':
            return state_dict_numpy[f'{prefix}encoder.layer.{num}.attention.output.dense.bias']
        elif name == f'layers.{num}.attn_norm.weight':
            if f'{prefix}encoder.layer.{num}.attention.output.LayerNorm.weight' in state_dict_numpy:
                return state_dict_numpy[f'{prefix}encoder.layer.{num}.attention.output.LayerNorm.weight']
            else:
                return state_dict_numpy[f'{prefix}encoder.layer.{num}.attention.output.LayerNorm.gamma']
        elif name == f'layers.{num}.attn_norm.bias':
            if f'{prefix}encoder.layer.{num}.attention.output.LayerNorm.bias' in state_dict_numpy:
                return state_dict_numpy[f'{prefix}encoder.layer.{num}.attention.output.LayerNorm.bias']
            else:
                return state_dict_numpy[f'{prefix}encoder.layer.{num}.attention.output.LayerNorm.beta']
        elif name == f'layers.{num}.ff.linear_hidden.0.weight':
            return state_dict_numpy[f'{prefix}encoder.layer.{num}.intermediate.dense.weight']
        elif name == f'layers.{num}.ff.linear_hidden.0.bias':
            return state_dict_numpy[f'{prefix}encoder.layer.{num}.intermediate.dense.bias']
        elif name == f'layers.{num}.ff.linear_out.weight':
            return state_dict_numpy[f'{prefix}encoder.layer.{num}.output.dense.weight']
        elif name == f'layers.{num}.ff.linear_out.bias':
            return state_dict_numpy[f'{prefix}encoder.layer.{num}.output.dense.bias']
        elif name == f'layers.{num}.ff_norm.weight':
            if f'{prefix}encoder.layer.{num}.output.LayerNorm.weight' in state_dict_numpy:
                return state_dict_numpy[f'{prefix}encoder.layer.{num}.output.LayerNorm.weight']
            else:
                return state_dict_numpy[f'{prefix}encoder.layer.{num}.output.LayerNorm.gamma']
        elif name == f'layers.{num}.ff_norm.bias':
            if f'{prefix}encoder.layer.{num}.output.LayerNorm.bias' in state_dict_numpy:
                return state_dict_numpy[f'{prefix}encoder.layer.{num}.output.LayerNorm.bias']
            else:
                return state_dict_numpy[f'{prefix}encoder.layer.{num}.output.LayerNorm.beta']
        else:
            raise ValueError(f'ERROR: Param "{name}" can not be loaded in Space Model!')


def hug2space(input_file, input_template, output_file):
    state_dict_output = OrderedDict()
    state_dict_input = torch.load(input_file, map_location=lambda storage, loc: storage)
    state_dict_template = torch.load(input_template, map_location=lambda storage, loc: storage)
    prefix = 'bert.' if list(state_dict_input.keys())[0].startswith('bert.') else ''

    for name, value in state_dict_template.items():
        match_value = get_match_value(name, state_dict_input, prefix)
        if match_value is not None:
            assert match_value.ndim == value.ndim
            if match_value.shape != value.shape:
                assert value.size(0) == BERT_VOCAB_SIZE and match_value.size(0) > BERT_VOCAB_SIZE
                match_value = match_value[:BERT_VOCAB_SIZE]
            dtype = value.dtype
            device = value.device
            state_dict_output[name] = torch.tensor(match_value, dtype=dtype, device=device)
        else:
            print(f'WARNING: Param "{name}" can not be loaded in Space Model.')

    torch.save(state_dict_output, output_file)


if __name__ == '__main__':
    # Using Example
    input_template = '../model/template.model'
    input_file = '../model/bert-base-uncased/pytorch_model.bin'
    output_file = '../model/bert-base-uncased.model'

    hug2space(input_file=input_file, input_template=input_template, output_file=output_file)
    print(f'Converted!')

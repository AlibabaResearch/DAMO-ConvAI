import os
import torch
from collections import OrderedDict


def get_match_value(name, state_dict_numpy):
    """
    Need be overridden towards different models, here for UnifiedTransformer Model
    """
    if name == 'bert.embeddings.word_embeddings.weight':
        return state_dict_numpy['embedder.token_embedding.weight']
    elif name == 'bert.embeddings.position_embeddings.weight':
        return state_dict_numpy['embedder.pos_embedding.weight']
    elif name == 'bert.embeddings.token_type_embeddings.weight':
        return state_dict_numpy['embedder.type_embedding.weight']
    elif name == 'bert.embeddings.LayerNorm.weight':
        return state_dict_numpy['embed_layer_norm.weight']
    elif name == 'bert.embeddings.LayerNorm.bias':
        return state_dict_numpy['embed_layer_norm.bias']
    elif name == 'bert.pooler.dense.weight':
        # return state_dict_numpy['pooler.0.weight']
        return None
    elif name == 'bert.pooler.dense.bias':
        # return state_dict_numpy['pooler.0.bias']
        return None
    elif name == 'cls.predictions.transform.dense.weight':
        return state_dict_numpy['mlm_transform.0.weight']
    elif name == 'cls.predictions.transform.dense.bias':
        return state_dict_numpy['mlm_transform.0.bias']
    elif name == 'cls.predictions.transform.LayerNorm.weight':
        return state_dict_numpy['mlm_transform.2.weight']
    elif name == 'cls.predictions.transform.LayerNorm.bias':
        return state_dict_numpy['mlm_transform.2.bias']
    elif name == 'cls.predictions.bias':
        return state_dict_numpy['mlm_bias']
    elif name == 'cls.predictions.decoder.weight':
        return state_dict_numpy['embedder.token_embedding.weight']
    elif name == 'cls.predictions.decoder.bias':
        return state_dict_numpy['mlm_bias']
    else:
        num = name.split('.')[3]
        assert num in [str(i) for i in range(12)]
        if name == f'bert.encoder.layer.{num}.attention.self.query.weight':
            qkv_weight = state_dict_numpy[f'layers.{num}.attn.linear_qkv.weight']
            return qkv_weight[:768]
        elif name == f'bert.encoder.layer.{num}.attention.self.key.weight':
            qkv_weight = state_dict_numpy[f'layers.{num}.attn.linear_qkv.weight']
            return qkv_weight[768: 1536]
        elif name == f'bert.encoder.layer.{num}.attention.self.value.weight':
            qkv_weight = state_dict_numpy[f'layers.{num}.attn.linear_qkv.weight']
            return qkv_weight[1536:]
        elif name == f'bert.encoder.layer.{num}.attention.self.query.bias':
            qkv_bias = state_dict_numpy[f'layers.{num}.attn.linear_qkv.bias']
            return qkv_bias[:768]
        elif name == f'bert.encoder.layer.{num}.attention.self.key.bias':
            qkv_bias = state_dict_numpy[f'layers.{num}.attn.linear_qkv.bias']
            return qkv_bias[768: 1536]
        elif name == f'bert.encoder.layer.{num}.attention.self.value.bias':
            qkv_bias = state_dict_numpy[f'layers.{num}.attn.linear_qkv.bias']
            return qkv_bias[1536:]
        elif name == f'bert.encoder.layer.{num}.attention.output.dense.weight':
            return state_dict_numpy[f'layers.{num}.attn.linear_out.weight']
        elif name == f'bert.encoder.layer.{num}.attention.output.dense.bias':
            return state_dict_numpy[f'layers.{num}.attn.linear_out.bias']
        elif name == f'bert.encoder.layer.{num}.attention.output.LayerNorm.weight':
            return state_dict_numpy[f'layers.{num}.attn_norm.weight']
        elif name == f'bert.encoder.layer.{num}.attention.output.LayerNorm.bias':
            return state_dict_numpy[f'layers.{num}.attn_norm.bias']
        elif name == f'bert.encoder.layer.{num}.intermediate.dense.weight':
            return state_dict_numpy[f'layers.{num}.ff.linear_hidden.0.weight']
        elif name == f'bert.encoder.layer.{num}.intermediate.dense.bias':
            return state_dict_numpy[f'layers.{num}.ff.linear_hidden.0.bias']
        elif name == f'bert.encoder.layer.{num}.output.dense.weight':
            return state_dict_numpy[f'layers.{num}.ff.linear_out.weight']
        elif name == f'bert.encoder.layer.{num}.output.dense.bias':
            return state_dict_numpy[f'layers.{num}.ff.linear_out.bias']
        elif name == f'bert.encoder.layer.{num}.output.LayerNorm.weight':
            return state_dict_numpy[f'layers.{num}.ff_norm.weight']
        elif name == f'bert.encoder.layer.{num}.output.LayerNorm.bias':
            return state_dict_numpy[f'layers.{num}.ff_norm.bias']
        else:
            raise ValueError('No matched name in state_dict_numpy!')


def space2hug(input_template, input_pt, output_pt, restore=True):
    state_dict_pytorch = OrderedDict()
    state_dict_init_template = torch.load(input_template, map_location=lambda storage, loc: storage)
    state_dict_init_pytorch = torch.load(input_pt, map_location=lambda storage, loc: storage)
    if 'module.' in list(state_dict_init_pytorch.keys())[0]:
        new_model_state_dict = OrderedDict()
        for k, v in state_dict_init_pytorch.items():
            assert k[:7] == 'module.'
            new_model_state_dict[k[7:]] = v
        state_dict_init_pytorch = new_model_state_dict

    for name, value in state_dict_init_template.items():
        match_value = get_match_value(name, state_dict_init_pytorch)
        if match_value is not None:
            assert match_value.shape == value.shape
            assert match_value.dtype == value.dtype
            state_dict_pytorch[name] = match_value
        else:
            print(f'Parm {name} is not existed! Restore: [{restore}]')
            if restore:
                state_dict_pytorch[name] = value
            else:
                continue

    torch.save(state_dict_pytorch, output_pt)


if __name__ == '__main__':
    # Using Example
    input_template = '../model/bert-base-uncased/pytorch_model.bin'
    input_pt = '../model/SPACE.model'
    output_dir = '../model/SPACE'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_pt = os.path.join(output_dir, 'pytorch_model.bin')

    restore = False
    space2hug(input_template=input_template, input_pt=input_pt, output_pt=output_pt, restore=restore)
    print(f'Converted!')

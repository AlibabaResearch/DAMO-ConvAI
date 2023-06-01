from multiprocessing import current_process
import torch
import copy

def change_text_maxlen(state_dict, max_pos):
    current_max_pos, embed_size = state_dict["text_embeddings.position_embeddings.weight"].shape
    if max_pos > current_max_pos:

        new_pos_embed = state_dict["text_embeddings.position_embeddings.weight"].new_empty(max_pos, embed_size)
        # print(new_pos_embed.shape)

        k = 0
        step = current_max_pos
        while k < max_pos - 1:
            dlen = min(step , max_pos-k)
            new_pos_embed[k:(k + step)] = state_dict["text_embeddings.position_embeddings.weight"][:dlen]
            k += step
        # print(new_pos_embed.shape)
        state_dict["text_embeddings.position_embeddings.weight"] = new_pos_embed
        state_dict["text_embeddings.position_ids"] = torch.arange(max_pos).expand((1, -1))
    else :
        state_dict["text_embeddings.position_embeddings.weight"] = state_dict["text_embeddings.position_embeddings.weight"][:max_pos]
        state_dict["text_embeddings.position_ids"] = torch.arange(max_pos).expand((1, -1))
    return state_dict
    
def expert_state_load(state_dict):
    out_dict = {}
    for k, v in state_dict.items():
        out_dict[k] = v
        if ".mlp" in k or ".norm2" in k:
            new_iv = copy.deepcopy(v)
            new_cv = copy.deepcopy(v)
            new_tv = copy.deepcopy(v)
            new_gv = copy.deepcopy(v)
            image_part = k.replace("mlp", "image_mlp").replace("norm2", "image_norm")
            caps_part = k.replace("mlp", "caps_mlp").replace("norm2","caps_norm")
            text_part = k.replace("mlp", "sentence_mlp").replace("norm2", "sentence_norm")
            generation_part = k.replace("mlp", "generation_mlp").replace("norm2", "generation_norm")

            out_dict[image_part] = new_iv
            out_dict[caps_part] = new_cv
            out_dict[text_part] = new_tv
            out_dict[generation_part] = new_gv

    return out_dict

def resize_token_embedding(state_dict , new_vs):
    word_embeddings = state_dict["text_embeddings.word_embeddings.weight"]
    decoder_weight = state_dict["mlm_score.decoder.weight"]
    decoder_bias = state_dict["mlm_score.bias"]
    vs , hs = word_embeddings.shape
    if new_vs > vs :
        new_word_embeddings = word_embeddings.new_empty(new_vs , hs)
        new_decoder_weight = decoder_weight.new_empty(new_vs , hs)
        new_decoder_bias = decoder_bias.new_empty(new_vs)
        new_word_embeddings.normal_(mean=0.0, std=0.02)
        new_decoder_weight.normal_(mean=0.0, std=0.02)
        new_decoder_bias.fill_(0)
        new_word_embeddings[:vs,:] = word_embeddings[:,:]
        new_decoder_weight[:vs ,:] = decoder_weight[:,:] 
        new_decoder_bias[:vs] = decoder_bias[:]
        state_dict["text_embeddings.word_embeddings.weight"] = new_word_embeddings
        state_dict["mlm_score.decoder.weight"] = new_decoder_weight
        state_dict["mlm_score.bias"] = new_decoder_bias
    return state_dict
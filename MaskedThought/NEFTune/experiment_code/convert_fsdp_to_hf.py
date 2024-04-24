3"""Convert hf model to checkpoint consummable by fsdp"""
import argparse
from transformers.models.llama.modeling_llama import LlamaForCausalLM
import torch.distributed._shard.checkpoint as dist_cp
import torch
import transformers

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_path", type=str, default="llama/7B_sharded")
    parser.add_argument("--save_path", type=str, default="llama/7B_new_hf")
    parser.add_argument("--added_tokens", type=int, default=1, help="Number of tokens added to the model that need to be ")
    parser.add_argument("--config_path", type=str, default="llama/7B_hf")
    args = parser.parse_args()

    model_config = transformers.AutoConfig.from_pretrained(args.config_path)
    model = LlamaForCausalLM(model_config).bfloat16()
    if args.added_tokens > 0:
        model.resize_token_embeddings(model.config.vocab_size + args.added_tokens)

    state_dict = model.state_dict()
    dist_cp.load_state_dict(
        state_dict=state_dict,
        storage_reader=dist_cp.FileSystemReader(args.load_path),
        no_dist=True
    )
    model.load_state_dict(state_dict)
    model.save_pretrained(args.save_path)
    

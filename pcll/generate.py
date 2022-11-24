from mycvae.utils import *
from mycvae.model import *
import pickle
import os
import math
import torch
import torch.nn.functional as F
from torch.nn import DataParallel
import numpy as np
import argparse
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
from tqdm import tqdm
from tqdm import trange
import importlib
# import logging
import copy
# from data.util import *
from collections import Counter
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from rouge import Rouge
from dataset import get_datasets
from torch.utils.data import DataLoader
from dataset import PadBatchSeq, TASK2INFO


# devices = '2'
# os.environ["CUDA_VISIBLE_DEVICES"] = devices

def top_k_top_p_filtering(logits, top_k=100, top_p=0.95, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value

    return logits


def repeat_score(text, ngram=[3, 4, 5, 6]):
    ngram_list = []
    for ng in ngram:
        ngram_list.append([text[idx:idx + ng] for idx in range(len(text) - ng - 1)])

    max_occurs = []
    for ngrams in ngram_list:
        count_result = Counter([' '.join(n) for n in ngrams])
        try:
            max_occurs.append(
                max(count_result.values())
            )
        except:
            pass

    scores = [max_oc / ((len(text) / ngram[idx]) + ngram[idx]) for idx, max_oc in enumerate(max_occurs)]
    return max(scores) if len(scores) >= 1 else 1.0

def get_reverse_mask(x_len):
    mask = torch.arange(max(x_len)-1, -1, -1, device=x_len.device)[None, :] < x_len[:, None]  # [bs, max_len]
    return mask.bool()

def sample_sequence(model, tokenizer, length, batch_size=None, x_mask=None, x_tokens=None, x_lens=None, 
                    temperature=1, top_k=100, top_p=0.95,  sampling=True, only_decoder=False):
    device = 'cuda'
    x_tokens = x_tokens.to(device)
    
    if x_mask is not None: x_mask = x_mask.to(device) 
    if x_lens is not None: 
        x_reverse_mask = get_reverse_mask(x_lens) 
    else:
        x_reverse_mask = None
    eos_token = tokenizer.eos_token_id

    # mem = None
    # prev = torch.tensor([[eos_token]] * batch_size, dtype=torch.long, device=device)
    # prev = prev.view(batch_size, -1)

    with torch.no_grad():
        if not only_decoder:
            prior_mean, prior_logvar = model.encoder_prior(input_ids=x_tokens, attention_mask=x_mask)[:2]
            latent_mean, latent_logvar = prior_mean, prior_logvar
            z = model.reparameterize(latent_mean, latent_logvar)
            assert not torch.isnan(z).any(), 'training get nan z'
            add_attn=True
        else:
            z = None
            add_attn=False
        
        # Z: [bs, 768]
        # _, mem = model.transformer(input_ids=x_tokens[:, :-1], past=None, attention_mask=x_reverse_mask, representations=z) # x_tokens--prompt
        _, mem = model.transformer(input_ids=x_tokens[:, :-1], past=None, representations=z, add_attn=add_attn) # x_tokens--prompt

        prev = x_tokens[..., -2].view(batch_size, -1)
        # print('prev',prev,flush=True)
        output = prev
        probability = torch.tensor([], dtype=torch.float, device=device)
        if_end = torch.tensor([False] * batch_size, dtype=torch.bool, device=device)
        # print('z size',z.size(),flush=True)

        for i in range(length): #trange
            # print('prev', prev.size(),flush=True)
            # one_col = torch.tensor([[True]] * batch_size, dtype=torch.bool, device=device)
            # if x_reverse_mask is not None:
            #     x_reverse_mask = torch.cat((x_reverse_mask, one_col), dim=1)

            logits, mem = model.transformer(input_ids=prev, past=mem, representations=z, add_attn=add_attn)
            # logits, mem = model.transformer(input_ids=prev, past=mem, attention_mask=x_reverse_mask,representations=z)

            logits = model.lm_head(logits)

            logits = logits[:, -1, :] / temperature
            logits = top_k_top_p_filtering(logits, top_k, top_p)
            probs = F.softmax(logits, dim=-1)

            if sampling:
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                _, next_token = torch.topk(probs, k=1, dim=-1)

            probability = torch.cat((probability, probs.gather(1, next_token)), dim=1)
            output = torch.cat((output, next_token), dim=1)
            prev = next_token

            # early stopping if all sents have ended once
            if_end[next_token.view(-1).eq(eos_token)] = True
            if if_end.all(): break

    return output[:,1:]
    # return output


def decode_text(text, eos, tokenizer):
    text = text[text.index(eos) + 1:]
    if eos in text:
        idx = text.index(eos)
        text = text[:idx]
    text=tokenizer.decode(text).strip()
    
    return text


def run_model():
    parser = argparse.ArgumentParser()
   # CHANGED 
    parser.add_argument('--experiment', type=str)
    parser.add_argument("--data_dir", default="./PLL_DATA/", 
        type=str, help="The path to train/dev/test data files.")   # Default parameters are set based on single GPU training
    parser.add_argument('--tasks',nargs='+', default=['banking'])
    parser.add_argument('--model_path', type=str, help='pretrained model path to local checkpoint')
   # NON-CHANGED 
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--length", type=int, default=-1)
    parser.add_argument("--temperature", type=int, default=0.95)
    parser.add_argument('--top_p', type=float, default=0.95)
    parser.add_argument('--top_k', type=int, default=100)
    parser.add_argument('--output_dir', type=str, default='out')

    parser.add_argument('--data_type', type=str, default='t1', choices=['t' + str(i) for i in range(9)], help="t: type")
    parser.add_argument('--model_type', type=str, default='cvae', choices=['cvae', 'ae_vae_fusion'])

   # use GPU
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--no_gpu', action="store_true")

    parser.add_argument('--add_input', action="store_true")
    parser.add_argument('--add_attn', action="store_true")
    parser.add_argument('--add_softmax', action="store_true")
    parser.add_argument('--attn_proj_vary', action="store_true")

    parser.add_argument('--learn_prior', action="store_true")
    parser.add_argument("--eval_batch_size", default=100, type=int, help="Batch size per GPU/CPU for evaluation.")

    args = parser.parse_args()
    args.log_file = os.path.join(args.output_dir, 'log.txt')       
    print(args)

    args.model_type = 'cvae'
    args.learn_prior = True
    
    # GPU
    if not torch.cuda.is_available(): args.no_gpu = True
    gpu = not args.no_gpu
    if gpu: torch.cuda.set_device(args.gpu)
    device = torch.device(args.gpu if gpu else "cpu")

    # randomness
    np.random.seed(args.seed)
    prng = np.random.RandomState()
    torch.random.manual_seed(args.seed)
    if gpu: torch.cuda.manual_seed(args.seed)

    if args.batch_size == -1:
        args.batch_size = 1

    # logging
    save_folder = os.path.join(args.output_dir, args.experiment)
    os.makedirs(save_folder, exist_ok=True)

    # importlib.reload(logging)
    logger = get_logger(args.log_file)
    logger.info('\n----------------------------------------------------------------------')

    print('Loading models...')
    cache_dir = os.path.join(args.output_dir, 'model_cache')
    os.makedirs(cache_dir, exist_ok=True)
    # Load pre-trained teacher tokenizer (vocabulary)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2', cache_dir=cache_dir)
    tokenizer.max_len = int(1e12)
    gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2', cache_dir=cache_dir)
    print('gpt2_params:', num_params(gpt2_model))  # gpt2: 124439808
    config = GPT2Config()

   # * Load VAE model 
    VAE = VAEModel(config, add_input=args.add_input, add_attn=args.add_attn, add_softmax=args.add_softmax,
                   attn_proj_vary=args.attn_proj_vary, learn_prior=args.learn_prior)

    args.load = args.model_path
    print('Loading model weights...')
    state = torch.load(args.load)
    VAE.load_state_dict(state)
    print('='*25,'Load trained model successfully.','='*25,flush=True)
    gc.collect()
    print('VAE_params:', num_params(VAE))  # 286694400

    print('Setup data...',flush=True)
    seq_len = VAE.config.n_ctx
    VAE.config.n_ctx=100 # xiu setting.
    print('VAE config n_ctx: sequence length',seq_len,flush=True)

   # * Get test loader. 
    datasets = get_datasets(args.data_dir, args.tasks, tokenizer, num_workers=1, ctx_max_len=100)
    for task in datasets:
        # data_loaders[task] = {}
        # train_sampler = torch.utils.data.RandomSampler(datasets[task]['train'])
        # train_loader = DataLoader(datasets[task]['train'],batch_size=args.train_batch_size,sampler=train_sampler, 
            # num_workers=1, pin_memory=True, collate_fn=PadBatchSeq(tokenizer.eos_token_id))
        # val_loader = DataLoader(datasets[task]['val'],batch_size=args.eval_batch_size,sampler=None, 
            # num_workers=1, pin_memory=True, collate_fn=PadBatchSeq(tokenizer.eos_token_id))
        test_loader = DataLoader(datasets[task]['test'],batch_size=args.eval_batch_size,sampler=None, 
            num_workers=1, pin_memory=True, collate_fn=PadBatchSeq(tokenizer.eos_token_id))

    VAE.eval() # be careful about VAE.eval() vs VAE.train()
    VAE.to(device)

    logger.info('\n----------------------------------------------------------------------')
    logger.info("Testing loop. batches: %d" % len(test_loader))

    endoftext = tokenizer.convert_tokens_to_ids("<|endoftext|>")

    n_samples = 0
    bleu4_sum = 0.0
    rouge_scores_values_sum = [0.0] * 9

    model_type = args.model_type

    check_file = open(os.path.join(save_folder, 'check.txt'), 'w', encoding='utf8')
    with tqdm(total=len(test_loader)) as pbar:
        ITER = enumerate(test_loader)
        for i, data in ITER:
            x_mask = data['prompt_mask']
            x_tokens = data['prompt_id']
            # x_lens = data['prompt_lens']
            x_lens = torch.tensor([len(i)-1 for i in x_tokens],dtype=torch.long).to(device)

            length = args.length
            # Changed to train.py
            if length == -1:
                # length = VAE.config.n_ctx  - 1
                length = VAE.config.n_ctx - x_tokens.size(1) - 1
            # elif length > VAE.config.n_ctx  - 1:
            elif length > VAE.config.n_ctx - x_tokens.size(1) - 1:
                raise ValueError("Can't get samples longer than window size: %s" % VAE.config.n_ctx)

            target_tokens = data['input_id'][..., 1:].contiguous()
            eff_samples = []
            n, l = target_tokens.size()
            # storys = [tokenizer.decode(target_tokens[i, :]) for i in range(n)]
            # storys = [s[s.find("<|endoftext|>") + len("<|endoftext|>"):] for s in storys]
            # storys_str = [s[:s.find("<|endoftext|>") + len("<|endoftext|>")] if "<|endoftext|>" in s else s for s in storys]

            for _ in range(1):
                all_out = []
                sample_time=5 # Sample 5 times.
                for zz in range(sample_time): 
                    out = sample_sequence(
                        model=VAE,
                        tokenizer=tokenizer,
                        length=length,
                        batch_size=x_tokens.size()[0],
                        x_mask=x_mask,
                        x_tokens=x_tokens,
                        x_lens=x_lens,
                        temperature=args.temperature,
                        top_k=args.top_k,
                        top_p=args.top_p,
                    )
                    out = out.tolist()
                    all_out.append(out)
    
               # * Check latent code z ability.
                for ss in range(len(all_out[0])):
                    check_file.write('\n')
                    check_file.write('='*20+'SAMPLE: %d'%n_samples+'='*20) 
                    check_file.write('\n')
                    iii=1
                    for oout in all_out:
                        text = decode_text(oout[ss], endoftext, tokenizer)
                        check_file.write(str(iii)+'==:')
                        check_file.write(text)
                        check_file.write('\n')
                        iii+=1
                    check_file.write('-'*100)
                    check_file.flush()

                    # text = decode_text(out[ss], endoftext, tokenizer)
                    # check_file.write('-'*100)
                    # check_file.write('\n')
                    # check_file.write('SAMPLE: %d'%n_samples)
                    # check_file.write('='*2)
                    # check_file.write(':')
                    # check_file.write(text)
                    # check_file.write('\n')
                    # check_file.write('-'*100)
                    # check_file.flush()

  
                    #     tx = decode_text(oout[ss], endoftext, tokenizer)
                    #     # print(tx,flush=True)
                    #     check_file.write(str(iii))
                    #     check_file.write('====') 
                    #     check_file.write(tx)
                    #     check_file.write('\n')
                    #     iii+=1
                    n_samples+=1
                    eff_samples.append((text,0))

                    if n_samples>=100: # Only evaluate part of the test data.
                        break

                   # bleu, rouge score calculation 
                    # score for one long text, higher than 0.075 usually means repetition
                    # rep_score = repeat_score(text.split(), ngram=[3, 4, 5, 6, 7, 8])
                    # if rep_score > 0.075:
                    #     # print(rep_score)
                    #     continue

                    # try:
                    #     # check bleu
                    #     bleu4 = sentence_bleu([storys_str[i].split()], text, smoothing_function=SmoothingFunction().method7)

                    #     # check rouge
                    #     rouge = Rouge()
                    #     rouge_scores = rouge.get_scores(text, storys_str[i])
                    #     rouge_scores_values = [v for k in rouge_scores[0].keys() for v in rouge_scores[0][k].values()]

                    #     bleu4_sum += bleu4
                    #     rouge_scores_values_sum = [v1 + v2 for v1, v2 in zip(rouge_scores_values_sum, rouge_scores_values)]
                    #     n_samples += 1
                    # except:
                    #     bleu4 = 0.0
                    #     rouge_scores = [{'rouge-1': {'f': 0.0, 'p': 0.0, 'r': 0.0},
                    #                      'rouge-2': {'f': 0.0, 'p': 0.0, 'r': 0.0},
                    #                      'rouge-l': {'f': 0.0, 'p': 0.0, 'r': 0.0}}]
                   # Store generated sentences.
                    # eff_samples.append((text,0))
                    # eff_samples.append((text, bleu4, rouge_scores))

                # write samples to file
                # for i in range(len(eff_samples)):
                    # samples_file.write("=" * 50 + " SAMPLE " + str(i) + " " + "=" * 50)
                    # samples_file.write('\n' * 2)

                    # samples_file.write("=" * 40 + " Outlines  " + "=" * 40)
                    # samples_file.write('\n' * 2)
                    # samples_file.write(tokenizer.decode(x_tokens[i, :][x_mask[i, :] == 1].tolist()))
                    # samples_file.write('\n' * 2)
                    # samples_file.write("=" * 40 + " Story " + "=" * 40)
                    # samples_file.write('\n' * 2)
                    # samples_file.write(storys_str[i])
                    # samples_file.write('\n' * 2)

                    # samples_file.write("=" * 40 + " Generated " + "=" * 40)
                    # samples_file.write('\n' * 2)
                    # samples_file.write(eff_samples[i][0])
                    # samples_file.write('\n' * 1)
                    # samples_file.flush()

                logger.info('batch %d finished.'%n_samples)
                pbar.update(1)

    print('Test complete with %d samples.' % n_samples)
    logger.info("Test complete with %d samples."%n_samples)

    # bleu4 = round(bleu4_sum / n_samples, 3)
    # rouge_scores_values = [round(r / n_samples, 3) for r in rouge_scores_values_sum]
    # print(' bleu-4:', bleu4)
    # print(' rouge :', rouge_scores_values)
    # logger.info(' bleu-4: %f', bleu4)
    # logger.info(' rouge : %s', str(rouge_scores_values))


if __name__ == '__main__':
    run_model()

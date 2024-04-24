# NEFTune

[10/17/2023] NEFTune has been integrated into the Huggingface's TRL (Transformer Reinforcement Learning) library and HF Trainer . [See Annoucement](https://x.com/younesbelkada/status/1714283468790935687?s=20).

[11/25/2023] NEFTune has been integrated into [Ludwig.ai](https://ludwig.ai) for LLM fine-tuning. [See PR](https://github.com/ludwig-ai/ludwig/pull/3744).

Please see the [limitations](https://github.com/neelsjain/NEFTune#limitations) of our study below. Additionally, for generation, we suggest using greedy decoding with a repetition penalty of $1.2$. Note that without the repetition penalty that we have seen performance degrade and generations to degenerate.

Feel free to reach out to me as well (email at the bottom of the page 1 in the paper).

## Overview
In this paper, we propose to add random noise to the embedding vectors of the training data during the forward pass of fine-tuning. We show that this simple trick can improve the outcome of instruction fine-tuning, often by a large margin, with no additional compute or data overhead. <u>N</u>oisy <u>E</u>mbedding Instruction <u>F</u>ine <u>T</u>uning (NEFTune), while simple, has a strong impact on downstream conversational quality.  When a raw LLM like LLaMA-2-7B is finetuned with noisy embeddings with popular Alpaca dataset, its performance on *AlpacaEval* improves from 29.8\% to 64.7\% -- an impressive boost of around 35 percentage points. NEFTune leads to this surprising and large jump in performance on conversational tasks, maintaining performance on factual question answering baselines. Using noisy embeddings seems to be a free lunch for LLM fine-tuning. The paper can be found [here](https://arxiv.org/abs/2310.05914).
<p align="center">
<img src=imgs/AlpacaEval_Figue1.png width="80%">
</p>

Note: During training, we observed the training loss values may be very close. The loss histograms in the Analysis section in the paper are the loss values when the noise is turned off. This is where the difference should show up.

## Code
The easiest way to incorporate NEFTune into your training procedure is to rewrite the forward for the embedding. An example of one way to do this for LLaMA is provided below. Note different distributed training will require different implementations.

```
from torch.nn import functional as F

def NEFTune(model, noise_alpha=5)
    def noised_embed(orig_embed, noise_alpha):
        def new_func(x):
            # during training, we add noise to the embedding
            # during generation, we don't add noise to the embedding
            if model.training:
                embed_init = orig_embed(x)
                dims = torch.tensor(embed_init.size(1) * embed_init.size(2))
                mag_norm = noise_alpha/torch.sqrt(dims)
                return embed_init + torch.zeros_like(embed_init).uniform_(-mag_norm, mag_norm)
            else:
                return orig_embed(x)
        return new_func
    ##### NOTE: this is for a LLaMA model ##### 
    ##### For a different model, you need to change the attribute path to the embedding #####
    model.base_model.model.model.embed_tokens.forward = noised_embed(model.base_model.model.model.embed_tokens, noise_alpha)
    return model
```

The code we used to run the experiments can be found in the `experiment_code` folder.

## Limitations
Our study has several limitations. We adopt AlpacaEval as our central measure of instruction following ability for LLMs, which is subject to the biases of a single judge (GPT-4). Additionally, due to limited computing resources, we were not able to validate the success of NEFTune on larger 70B variants of LLaMA-2 on multiple datasets, and we had to rely on fixed hyper-parameters for most NEFTune runs rather than sweeping. Finally, despite our empirical studies, we do not have a conclusive understanding of why NEFTune works.

## Call for Feedback
Our study was limited to the settings that we explored. Please feel free to open an issue regarding any weakness of NEFTune. We hope to be open about any issues with NEFTune to help future research and users.

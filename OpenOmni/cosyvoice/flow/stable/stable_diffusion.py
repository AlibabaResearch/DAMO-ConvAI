import torch
from torch.nn import functional as F
from .dit import DiffusionTransformer
from .adp import UNet1d
from .sampling import sample
import math
from model.base import BaseModule
import pdb

target_length = 1536


def pad_and_create_mask(matrix, target_length):
    T = matrix.shape[2]
    if T > target_length:
        raise ValueError("The third dimension length %s should not exceed %s" % (T, target_length))

    padding_size = target_length - T

    padded_matrix = F.pad(matrix, (0, padding_size), "constant", 0)

    mask = torch.ones((1, target_length))
    mask[:, T:] = 0  # Set the padding part to 0

    return padded_matrix.to(matrix.device), mask.to(matrix.device)


class Stable_Diffusion(BaseModule):
    def __init__(self, io_channels, input_concat_dim=None, embed_dim=768, depth=24, num_heads=24,
                 project_cond_tokens=False, transformer_type="continuous_transformer"):
        super(Stable_Diffusion, self).__init__()
        self.diffusion = DiffusionTransformer(
            io_channels=io_channels,
            input_concat_dim=input_concat_dim,
            embed_dim=embed_dim,
            # cond_token_dim=target_length,
            depth=depth,
            num_heads=num_heads,
            project_cond_tokens=project_cond_tokens,
            transformer_type=transformer_type,
        )
        # self.diffusion = UNet1d(
        #                   in_channels=80,
        #                   channels=256,
        #                   resnet_groups=16,
        #                   kernel_multiplier_downsample=2,
        #                   multipliers=[4, 4, 4, 5, 5],
        #                   factors=[1, 2, 2, 4], # 输入长度不一致卷积缩短
        #                   num_blocks=[2, 2, 2, 2],
        #                   attentions=[1, 3, 3, 3, 3],
        #                   attention_heads=16,
        #                   attention_multiplier=4,
        #                   use_nearest_upsample=False,
        #                   use_skip_scale=True,
        #                   use_context_time=True
        #                   )
        self.rng = torch.quasirandom.SobolEngine(1, scramble=True)

    @torch.no_grad()
    def forward(self, mu, mask, n_timesteps):
        # pdb.set_trace()
        mask = mask.squeeze(1)
        noise = torch.randn_like(mu).to(mu.device)
        # mu_pad, mu_pad_mask = pad_and_create_mask(mu, target_length)
        # extra_args = {"cross_attn_cond": mu, "cross_attn_cond_mask": mask, "mask": mask}
        extra_args = {"input_concat_cond": mu, "mask": mask}
        fakes = sample(self.diffusion, noise, n_timesteps, 0, **extra_args)

        return fakes

    def compute_loss(self, x0, mask, mu):

        # pdb.set_trace()
        t = self.rng.draw(x0.shape[0])[:, 0].to(x0.device)
        alphas, sigmas = torch.cos(t * math.pi / 2), torch.sin(t * math.pi / 2)

        alphas = alphas[:, None, None]
        sigmas = sigmas[:, None, None]
        noise = torch.randn_like(x0)
        noised_inputs = x0 * alphas + noise * sigmas
        targets = noise * alphas - x0 * sigmas
        mask = mask.squeeze(1)
        # mu_pad, mu_pad_mask = pad_and_create_mask(mu, target_length)
        # output = self.diffusion(noised_inputs, t, cross_attn_cond=mu, 
        #                         cross_attn_cond_mask=mask, mask=mask, cfg_dropout_prob=0.1)
        # pdb.set_trace()
        output = self.diffusion(noised_inputs,  # [bs, 80, 229]
                                t,  # (bs,)
                                input_concat_cond=mu,
                                mask=mask,  # [bs, 229]
                                cfg_dropout_prob=0.1)

        return self.mse_loss(output, targets, mask), output

    def mse_loss(self, output, targets, mask):

        mse_loss = F.mse_loss(output, targets, reduction='none')

        if mask.ndim == 2 and mse_loss.ndim == 3:
            mask = mask.unsqueeze(1)

        if mask.shape[1] != mse_loss.shape[1]:
            mask = mask.repeat(1, mse_loss.shape[1], 1)

        mse_loss = mse_loss * mask

        mse_loss = mse_loss.mean()

        return mse_loss

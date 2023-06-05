import torch


class MomentumOptim:
    def __init__(self, step_size=0.01, momentum=0.9):
        self.step_size = step_size
        self.momentum = momentum
        self.m = None  # velocity

    def init(self):
        self.m = None

    def upd_m(self, old_m, g):
        return g + self.momentum * old_m

    def upd(self, old_x, m):
        return old_x + self.step_size * m

    def __call__(self, old_xs, new_xs):
        pesudo_gs = [new_x - old_x for old_x, new_x in zip(old_xs, new_xs)]

        if not self.m:
            self.m = pesudo_gs
        else:
            self.m = [self.upd_m(old_m, g) for old_m, g in zip(self.m, pesudo_gs)]

        updated_kv = [self.upd(old_x, m) for old_x, m in zip(old_xs, self.m)]
        return updated_kv


class AttnOptimWrapper:
    def __init__(self, llm, model_type, optimizer="momentum", **optimizer_args):
        self.model = llm
        self.kv = None
        self.model_type = model_type

        if optimizer == "momentum":
            self.optim_k = MomentumOptim(**optimizer_args)
            self.optim_v = MomentumOptim(**optimizer_args)
        else:
            raise ValueError()

    def init(self):
        self.optim_k.init()
        self.optim_v.init()

    @torch.no_grad()
    def step(self, ctx_ids):
        L = len(ctx_ids)

        ctx_ids = ctx_ids.unsqueeze(0)  # [1, L]
        mask = torch.ones_like(ctx_ids)
        if self.kv is not None:
            mask = mask.repeat(1, 2)  # [1, 2*L]

        next_kv = self.model(
            input_ids=ctx_ids,
            attention_mask=mask,
            past_key_values=self.kv,
            use_cache=True,
        ).past_key_values  # kv @ (old_ctx + new_ctx)

        cur_kv = []
        for layer_k, layer_v in next_kv:
            # [B, num_head, 2*L, head_hidden]
            cur_kv.append([layer_k[:, :, -L:, :], layer_v[:, :, -L:, :]])  # kv @ (new_ctx)

        if not self.kv:
            self.kv = cur_kv
        else:
            old_ks, old_vs = zip(*self.kv)
            cur_ks, cur_vs = zip(*cur_kv)

            upd_ks = self.optim_k(old_ks, cur_ks)
            upd_vs = self.optim_v(old_vs, cur_vs)
            self.kv = list(zip(upd_ks, upd_vs))

        return self.kv

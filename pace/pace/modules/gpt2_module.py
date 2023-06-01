import os
import torch
import torch.nn as nn
import pytorch_lightning as pl

from transformers import AutoConfig, AutoModelForCausalLM, get_linear_schedule_with_warmup
from transformers.optimization import AdamW

from pace.modules import heads, objectives, pace_utils

class DialGPT2(pl.LightningModule):
    def __init__(self, config, tokenizer):
        super().__init__()

        self.save_hyperparameters()

        self.config = AutoConfig.from_pretrained(
            config["model_config"], cache_dir=config["cache_dir"]
        )
        self.tokenizer = tokenizer
        self.transforms = AutoModelForCausalLM.from_pretrained(
            config["model_config"],
            from_tf=bool(".ckpt" in config["model_config"]),
            config=self.config,
            cache_dir=config["cache_dir"],
        )
        if config['add_special_tokens']:
            self.transforms.resize_token_embeddings(len(tokenizer))

        pace_utils.set_metrics(self)
        self.current_tasks = list()

        if self.hparams.config["load_path"] != "" and self.hparams.config["test_only"]:
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            self.load_state_dict(state_dict, strict=False)
    
    def forward(self, batch):
        # NO MLM NOW
        inputs, labels = batch["input_ids"], batch["input_ids"]
        outputs = self.transforms(inputs, labels=labels)
        ret = {
            "rg_loss": outputs[0]
        }

        phase = "train" if self.training else "val"
        loss = getattr(self, f"{phase}_rg_loss")(ret["rg_loss"])
        self.log(f"rg/{phase}/loss", loss)

        return ret


    def training_step(self, batch, batch_idx):
        pace_utils.set_task(self)
        output = self(batch)
        total_loss = sum([v for k, v in output.items() if "loss" in k])

        return total_loss

    def training_epoch_end(self, outs):
        phase = "train" if self.training else "val"
        the_metric = 0
        for loss_name, v in self.hparams.config["loss_names"].items():
            if v < 1:
                continue

            value = 0
            if loss_name == "rg":
                value = - getattr(self, f"{phase}_{loss_name}_loss").compute()
                self.log(f"{loss_name}/{phase}/loss_epoch", value)
                getattr(self, f"{phase}_{loss_name}_loss").reset()
            the_metric += value

        self.log(f"{phase}/the_metric", the_metric)

    def validation_step(self, batch, batch_idx):
        pace_utils.set_task(self)
        output = self(batch)

    def validation_epoch_end(self, outs):
        phase = "train" if self.training else "val"
        the_metric = 0
        for loss_name, v in self.hparams.config["loss_names"].items():
            if v < 1:
                continue

            value = 0
            if loss_name == "rg":
                value = - getattr(self, f"{phase}_{loss_name}_loss").compute()
                self.log(f"{loss_name}/{phase}/loss_epoch", value)
                getattr(self, f"{phase}_{loss_name}_loss").reset()
            the_metric += value

        self.log(f"{phase}/the_metric", the_metric)

    def test_step(self, batch, batch_idx):
        pace_utils.set_task(self)
        output = self(batch)
        ret = dict()
        if self.hparams.config["loss_names"]["rg"] > 0:
            ret.update(objectives.rg_test_step(self, batch))
        return ret

    def test_epoch_end(self, outs):
        model_name = self.hparams.config["datasets"][0]
        ret = objectives.rg_test_wrapup(outs, model_name)
        pace_utils.epoch_wrapup(self)

    def configure_optimizers(self):
        lr = self.hparams.config["learning_rate"]
        wd = self.hparams.config["weight_decay"]
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.transforms.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": wd,
            },
            {
                "params": [
                    p
                    for n, p in self.transforms.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters, lr=lr, eps=1e-8
        )

        if self.trainer.max_steps is None:
            max_steps = (
                len(self.trainer.datamodule.train_dataloader())
                * self.trainer.max_epochs
                // self.trainer.accumulate_grad_batches
            )
        else:
            max_steps = self.trainer.max_steps

        warmup_steps = self.hparams.config["warmup_steps"]
        if isinstance(self.hparams.config["warmup_steps"], float):
            warmup_steps = int(max_steps * warmup_steps)

        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=max_steps
        )
        model_name_or_path = self.hparams.config["model_config"]
        if (
            model_name_or_path
            and os.path.isfile(os.path.join(model_name_or_path, "optimizer.pt"))
            and os.path.isfile(os.path.join(model_name_or_path, "scheduler.pt"))
        ):
            # Load in optimizer and scheduler states
            optimizer.load_state_dict(
                torch.load(os.path.join(model_name_or_path, "optimizer.pt"))
            )
            scheduler.load_state_dict(
                torch.load(os.path.join(model_name_or_path, "scheduler.pt"))
            )

        sched = {"scheduler": scheduler, "interval": "step"}
        return (
            [optimizer],
            [sched],
        )





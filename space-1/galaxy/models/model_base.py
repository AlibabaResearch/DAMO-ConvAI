"""
Model base
"""
import torch
import torch.nn as nn


class ModelBase(nn.Module):
    """
    Basic model wrapper.
    """
    _registry = dict()

    @classmethod
    def register(cls, name):
        ModelBase._registry[name] = cls
        return

    @staticmethod
    def by_name(name):
        return ModelBase._registry[name]

    @staticmethod
    def create(hparams, *args, **kwargs):
        model_cls = ModelBase.by_name(hparams.model)
        return model_cls(hparams, *args, **kwargs)

    @classmethod
    def add_cmdline_argument(cls, parser):
        """ Add cmdline argument. """
        group = parser.add_argument_group("Model")
        group.add_argument("--init_checkpoint", type=str, default=None)
        group.add_argument("--model", type=str, default="UnifiedTransformer",
                           choices=["UnifiedTransformer", "PretrainUnifiedTransformer"])
        args, _ = parser.parse_known_args()
        model_cls = ModelBase.by_name(args.model)
        model_cls.add_cmdline_argument(group)
        return group

    def __init__(self, hparams):
        super(ModelBase, self).__init__()
        self.init_checkpoint = hparams.init_checkpoint
        self.with_rdrop_act = hparams.with_rdrop_act
        self.use_gpu = hparams.use_gpu
        return

    def _create_parameters(self):
        """ Create model's paramters. """
        raise NotImplementedError

    def _forward(self, inputs, is_training):
        """ Real forward process of model in different mode(train/test). """
        raise NotImplementedError

    def _collect_metrics(self, inputs, outputs):
        """ Calculate loss function by using inputs and outputs. """
        raise NotImplementedError

    def _optimize(self, loss, do_update, optimizer):
        """ Optimize loss function and update model. """
        raise NotImplementedError

    def _infer(self, inputs, start_id, eos_id, max_gen_len, prev_input):
        """ Real inference process of model. """
        raise NotImplementedError

    def forward(self, inputs, is_training=False):
        """
        Forward process, include real forward, collect metrics.

        @params : inputs : input data
        @type : dict of numpy.ndarray/int/float/...
        """
        if is_training:
            self.train()
        else:
            self.eval()

        if self.with_rdrop_act:
            inputs = {k: torch.cat([v, v], dim=0) if isinstance(v, torch.Tensor) else v
                      for k, v in inputs.items()}

        outputs = self._forward(inputs, is_training)
        metrics = self._collect_metrics(inputs, outputs)

        return metrics

    def infer(self, inputs, start_id=None, eos_id=None, max_gen_len=None, prev_input=None):
        """
        Inference process.

        @params : inputs : input data
        @type : dict of numpy.ndarray/int/float/...
        """
        self.eval()
        outputs = self._infer(inputs, start_id=start_id, eos_id=eos_id,
                              max_gen_len=max_gen_len, prev_input=prev_input)
        return outputs

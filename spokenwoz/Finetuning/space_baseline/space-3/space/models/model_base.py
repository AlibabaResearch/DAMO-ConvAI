"""
Model base
"""
import torch.nn as nn


class ModelBase(nn.Module):
    """
    Basic model wrapper for static graph and dygrpah.
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
                           choices=["UnifiedTransformer", "IntentUnifiedTransformer",
                                    "GenUnifiedTransformer"])
        args, _ = parser.parse_known_args()
        model_cls = ModelBase.by_name(args.model)
        model_cls.add_cmdline_argument(group)
        return group

    def __init__(self, hparams):
        super(ModelBase, self).__init__()
        self.init_checkpoint = hparams.init_checkpoint
        self.abandon_label = hparams.abandon_label
        self.use_gpu = hparams.use_gpu
        self.gpu = hparams.gpu
        return

    def _create_parameters(self):
        """ Create model's paramters. """
        raise NotImplementedError

    def _forward(self, inputs, is_training, with_label):
        """ NO LABEL: Real forward process of model in different mode(train/test). """
        raise NotImplementedError

    def _collect_metrics(self, inputs, outputs, with_label, data_file):
        """ NO LABEL: Calculate loss function by using inputs and outputs. """
        raise NotImplementedError

    def _optimize(self, loss, optimizer, lr_scheduler):
        """ Optimize loss function and update model. """
        raise NotImplementedError

    def _infer(self, inputs, start_id, eos_id, max_gen_len, prev_input):

        """ Real inference process of model. """
        raise NotImplementedError

    def forward(self, inputs, is_training=False, with_label=False, data_file=None):
        """
        Forward process, include real forward, collect metrices and optimize(optional)

        @params : inputs : input data
        @type : dict of numpy.ndarray/int/float/...
        """
        if is_training:
            self.train()
        else:
            self.eval()

        with_label = False if self.abandon_label else with_label
        outputs = self._forward(inputs, is_training, with_label=with_label)
        metrics = self._collect_metrics(inputs, outputs, with_label=with_label, data_file=data_file)

        return metrics

    def infer(self, inputs, start_id=None, eos_id=None, max_gen_len=None, prev_input=None):
        """
        Inference process.

        @params : inputs : input data
        @type : dict of numpy.ndarray/int/float/...
        """
        self.eval()
        results = self._infer(inputs, start_id=start_id, eos_id=eos_id,
                              max_gen_len=max_gen_len, prev_input=prev_input)
        return results

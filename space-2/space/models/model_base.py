"""
Model base
"""
import torch
import torch.nn as nn


class ModelBase(nn.Module):
    """
    Basic model wrapper for static graph and dygrpah.

    _registry, register, by_name, create用于管理不同的子类（具体模型）
    具体的模型继承父类ModelBase，使用其register方法将子类注册到父类的_registry属性中，用于父类管理所有的子类
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
                           choices=["UnifiedTransformer", "IntentUnifiedTransformer"])
        args, _ = parser.parse_known_args()
        model_cls = ModelBase.by_name(args.model)
        model_cls.add_cmdline_argument(group)
        return group

    def __init__(self, hparams):
        super(ModelBase, self).__init__()
        self.init_checkpoint = hparams.init_checkpoint
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

    def _infer(self, inputs):
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

        outputs = self._forward(inputs, is_training, with_label=with_label)
        metrics = self._collect_metrics(inputs, outputs, with_label=with_label, data_file=data_file)

        return metrics

    def infer(self, inputs):
        """
        Inference process.

        @params : inputs : input data
        @type : dict of numpy.ndarray/int/float/...
        """
        self.eval()
        results = self._infer(inputs)
        results = {name: results[name].cpu().detach().numpy() for name in results}
        return results

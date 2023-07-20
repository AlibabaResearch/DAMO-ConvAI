import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss


def compute_kl_loss(p, q, filter_scores=None):
    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')

    # You can choose whether to use function "sum" and "mean" depending on your task
    p_loss = p_loss.sum(dim=-1)
    q_loss = q_loss.sum(dim=-1)

    # mask is for filter mechanism
    if filter_scores is not None:
        p_loss = filter_scores * p_loss
        q_loss = filter_scores * q_loss

    p_loss = p_loss.mean()
    q_loss = q_loss.mean()

    loss = (p_loss + q_loss) / 2
    return loss


class CatKLLoss(_Loss):
    """
    CatKLLoss
    """
    def __init__(self, reduction='mean'):
        super(CatKLLoss, self).__init__()
        assert reduction in ['none', 'sum', 'mean']
        self.reduction = reduction

    def forward(self, log_qy, log_py):
        """
        KL(qy|py) = Eq[qy * log(q(y) / p(y))]

        log_qy: (batch_size, latent_size)
        log_py: (batch_size, latent_size)
        """
        qy = torch.exp(log_qy)
        kl = torch.sum(qy * (log_qy - log_py), dim=1)

        if self.reduction == 'mean':
            kl = kl.mean()
        elif self.reduction == 'sum':
            kl = kl.sum()
        return kl


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = features.device if features.is_cuda else torch.device('cpu')

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float()
        else:
            mask = mask.float()

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


if __name__ == '__main__':
    # define hyper-parameters
    temp = 0.07
    batch_size = 8
    n_views = 2
    n_class = 5
    hidden_dim = 64

    # define loss with a temperature `temp`
    criterion = SupConLoss(temperature=temp)

    # features: [bsz, n_views, f_dim]
    # `n_views` is the number of crops from each image
    # better be L2 normalized in f_dim dimension
    features = torch.rand(batch_size, n_views, hidden_dim)
    features = F.normalize(features, dim=-1, p=2)

    # labels: [bsz]
    labels = torch.randint(0, n_class, (batch_size,))

    # 1. supervised contrastive loss (SupContrast)
    loss_sup = criterion(features, labels)
    print(loss_sup)

    # 2. self-supervised contrastive loss (SimCLR)
    loss_self = criterion(features)
    print(loss_self)

    # 3. soft supervised contrastive loss (Space)
    # mask: contrastive mask of shape [bsz, bsz],
    # mask_{i,j} = sim_score where indicates the similarity score between sample j and sample i.
    # Can be asymmetric.
    # Using: loss_sup = criterion(features, mask=mask)  # no need for labels anymore

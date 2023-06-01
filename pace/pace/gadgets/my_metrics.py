import torch
from pytorch_lightning.metrics import Metric

def scores_to_ranks(scores: torch.Tensor):
    """Convert model output scores into ranks."""
    batch_size, num_rounds , num_options = scores.size()
    scores = scores.view(-1, num_options)

    # sort in descending order - largest score gets highest rank
    sorted_ranks, ranked_idx = scores.sort(1, descending=True)

    # i-th position in ranked_idx specifies which score shall take this
    # position but we want i-th position to have rank of score at that
    # position, do this conversion
    ranks = ranked_idx.clone().fill_(0)
    for i in range(ranked_idx.size(0)):
        for j in range(num_options):
            ranks[i][ranked_idx[i][j]] = j
    # convert from 0-99 ranks to 1-100 ranks
    ranks += 1
    ranks = ranks.view(batch_size, num_rounds, num_options)
    return ranks

class Accuracy(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("correct", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, logits, target):
        logits, target = (
            logits.detach().to(self.correct.device),
            target.detach().to(self.correct.device),
        )
        preds = logits.argmax(dim=-1)
        preds = preds[target != -100]
        target = target[target != -100]
        if target.numel() == 0:
            return 1

        assert preds.shape == target.shape

        self.correct += torch.sum(preds == target)
        self.total += target.numel()

    def compute(self):
        return self.correct / self.total


class Scalar(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("scalar", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, scalar):
        if isinstance(scalar, torch.Tensor):
            scalar = scalar.detach().to(self.scalar.device)
        else:
            scalar = torch.tensor(scalar).float().to(self.scalar.device)
        self.scalar += scalar
        self.total += 1

    def compute(self):
        return self.scalar / self.total


class VQAScore(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("score", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, logits, target):
        logits, target = (
            logits.detach().float().to(self.score.device),
            target.detach().float().to(self.score.device),
        )
        logits = torch.max(logits, 1)[1]
        one_hots = torch.zeros(*target.size()).to(target)
        one_hots.scatter_(1, logits.view(-1, 1), 1)
        scores = one_hots * target

        self.score += scores.sum()
        self.total += len(logits)

    def compute(self):
        return self.score / self.total

class MMConvDSTScore(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("joint", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("ga", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("os", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("sl", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("ac", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("pos", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def add_one(self, name):
        origin_value = getattr(self, name)
        setattr(self, name, origin_value + 1) 

    def update(self, results):
        for id_, result in results.items():
            for sample in result:
                if sample:
                    self.add_one('total')
                    joint_correct = True
                    # 如果gate都预测错了，那相当于后面的怎么预测都是错的
                    if sample['predictions']['ga'] != sample['gts']['ga']:
                        self.add_one('pos')
                        continue
                    else: # 否则ga加一
                        self.add_one('ga')
                        if sample['gts']['ga']==0:
                            pass
                            # scores['joint'] += 1 # 其他scores不计入计算
                        else:
                            self.add_one('pos')
                            for target in ['os','sl','ac']:
                                if sample['predictions'][target] == sample['gts'][target]:
                                    self.add_one(target)
                                else:
                                    joint_correct = False
                            if joint_correct:
                                self.add_one('joint')

    def compute(self):
        ret = {
            "ga": self.ga / self.total,
            "os": self.os / self.pos,
            "ac": self.ac / self.pos,
            "sl": self.sl / self.pos,
            "joint": self.joint / self.pos
        }
        return ret

class NDCG(object):
    def __init__(self):
        self._ndcg_numerator = 0.0
        self._ndcg_denominator = 0.0

    def observe(
        self, predicted_scores: torch.Tensor, target_relevance: torch.Tensor
    ):
        """
        Observe model output scores and target ground truth relevance and
        accumulate NDCG metric.

        Parameters
        ----------
        predicted_scores: torch.Tensor
            A tensor of shape (batch_size, num_options), because dense
            annotations are available for 1 randomly picked round out of 10.
        target_relevance: torch.Tensor
            A tensor of shape same as predicted scores, indicating ground truth
            relevance of each answer option for a particular round.
        """
        predicted_scores = predicted_scores.detach()

        # shape: (batch_size, 1, num_options)
        predicted_scores = predicted_scores.unsqueeze(1)
        predicted_ranks = scores_to_ranks(predicted_scores)

        # shape: (batch_size, num_options)
        predicted_ranks = predicted_ranks.squeeze()
        batch_size, num_options = predicted_ranks.size()
        
        k = torch.sum(target_relevance != 0, dim=-1)

        # shape: (batch_size, num_options)
        _, rankings = torch.sort(predicted_ranks, dim=-1)
        # Sort relevance in descending order so highest relevance gets top rnk.
        _, best_rankings = torch.sort(
            target_relevance, dim=-1, descending=True
        )

        # shape: (batch_size, )
        batch_ndcg = []
        for batch_index in range(batch_size):
            num_relevant = k[batch_index]
            dcg = self._dcg(
                rankings[batch_index][:num_relevant],
                target_relevance[batch_index],
            )
            best_dcg = self._dcg(
                best_rankings[batch_index][:num_relevant],
                target_relevance[batch_index],
            )
            batch_ndcg.append(dcg / best_dcg)

        self._ndcg_denominator += batch_size
        self._ndcg_numerator += sum(batch_ndcg)

    def _dcg(self, rankings: torch.Tensor, relevance: torch.Tensor):
        sorted_relevance = relevance[rankings].cpu().float()
        discounts = torch.log2(torch.arange(len(rankings)).float() + 2)
        return torch.sum(sorted_relevance / discounts, dim=-1)

    def retrieve(self, reset: bool = True):
        if self._ndcg_denominator > 0:
            metrics = {
                "ndcg": float(self._ndcg_numerator / self._ndcg_denominator)
            }
        else:
            metrics = {}

        if reset:
            self.reset()
        return metrics

    def reset(self):
        self._ndcg_numerator = 0.0
        self._ndcg_denominator = 0.0
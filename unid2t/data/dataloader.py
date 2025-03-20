import torch

from torch.utils.data import DataLoader, DistributedSampler


def init_dataloader(dataset, batch_size=16, num_workers=5, dist_train=False, pin_memory=False, shuffle=True, sampler=None):
    """
    sampler = None
    if dist_train:
        shuffle = False
        sampler = DistributedSampler(dataset)
    """

    dataloader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            shuffle=shuffle, sampler=sampler,
                            batch_sampler=None,
                            num_workers=num_workers,
                            collate_fn=dataset.collate_fn,
                            pin_memory=pin_memory)

    return dataloader


import torch

def pad_and_tensorize_sequence(sequences, padding_value = None, tensorize = False):
    if tensorize:
        return torch.tensor(sequences, dtype=torch.long)
    max_size = max([len(sequence) for sequence in sequences])
    padded_sequences = []
    for sequence in sequences:
        padded_sequence = sequence + [padding_value] * (max_size - len(sequence))
        padded_sequences.append(padded_sequence)
    return torch.tensor(padded_sequences, dtype=torch.long)


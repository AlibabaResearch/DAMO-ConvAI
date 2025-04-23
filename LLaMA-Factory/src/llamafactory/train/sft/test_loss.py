import torch
import torch.nn.functional as F

IGNORE_INDEX = -100

def reward_weighted_loss(nll_loss, scores):
    nll_loss = nll_loss.squeeze(-1)  # Size: [batch_size, seq_length]
    batch_size = nll_loss.size(0)

    final_weighted_loss = torch.zeros_like(nll_loss)

    for i in range(batch_size):
        current_nll_loss = nll_loss[i]

        # Identify non-zero segments
        non_zero_mask = current_nll_loss != 0
        non_zero_indices = non_zero_mask.nonzero(as_tuple=False).squeeze(-1)

        if non_zero_indices.numel() == 0:
            continue

        # Detect the start of each segment
        segment_boundaries = non_zero_indices[
            torch.cat((torch.tensor([True], device=non_zero_indices.device), 
                       non_zero_indices[1:] != (non_zero_indices[:-1] + 1))).nonzero(as_tuple=False).squeeze(-1)
        ]

        # Identify the ends of segments
        segment_ends = torch.cat((segment_boundaries[1:], torch.tensor([current_nll_loss.size(0)], device=segment_boundaries.device)))
        segment_lengths = segment_ends - segment_boundaries

        # Extract valid scores
        valid_scores = scores[i][scores[i] != IGNORE_INDEX]
        num_valid_scores = valid_scores.size(0)

        assert num_valid_scores == len(segment_lengths), \
            f"Mismatch in number of valid scores and non-zero segments: {num_valid_scores} vs {len(segment_lengths)}"

        # Apply weights to each segment
        for j in range(num_valid_scores):
            start = segment_boundaries[j].item()
            length = segment_lengths[j].item()

            final_weighted_loss[i, start:start + length] = current_nll_loss[start:start + length] * valid_scores[j]

    return final_weighted_loss

# Test with given example tensors
torch.manual_seed(0)  # For reproducibility

seq_length = 1439
num_scores = 10

# Simulated nll_loss (batch_size=2, seq_length=1439)
nll_loss = torch.zeros(2, seq_length, 1)
nll_loss[0, 10:20] = 0.5  # First non-zero segment
nll_loss[0, 50:60] = 0.3  # Second non-zero segment
nll_loss[0, 200:210] = 0.2  # Third non-zero segment
nll_loss[0, 400:420] = 0.7  # Fourth non-zero segment
nll_loss[0, 1000:1010] = 0.4  # Fifth non-zero segment

# Simulated scores (batch_size=2, num_scores=10)
scores = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, -100.0, -100.0, -100.0, -100.0, -100.0],
                       [0.1, 0.2, 0.3, 0.4, 0.5, -100.0, -100.0, -100.0, -100.0, -100.0]])

# Execute function
weighted_loss = reward_weighted_loss(nll_loss, scores)
print(weighted_loss.size())
print(weighted_loss)

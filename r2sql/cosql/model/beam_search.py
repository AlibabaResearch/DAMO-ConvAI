import heapq
import math
import numpy as np
from typing import Callable, Dict, Tuple, List
import torch

class BeamSearchState:
    def __init__(self, sequence=[], log_probability=0, state={}):
        self.sequence = sequence
        self.log_probability = log_probability
        self.state = state

    def __lt__(self, other):
        return self.log_probability < other.log_probability

    def __str__(self):
        return f"{self.log_probability}: {' '.join(self.sequence)}"

class BeamSearch:
    """
    Implements the beam search algorithm for decoding the most likely sequences.
    """
    def __init__(
        self,
        is_end_of_sequence: Callable[[int, str], bool],
        max_steps: int = 50,
        beam_size: int = 10,
        per_node_beam_size: int = None,
    ) -> None:
        self.is_end_of_sequence = is_end_of_sequence
        self.max_steps = max_steps
        self.beam_size = beam_size
        self.per_node_beam_size = per_node_beam_size
        if not per_node_beam_size:
            self.per_node_beam_size = beam_size

        self.beam = []

    def append(self, log_probability, new_state):
        if len(self.beam) < self.beam_size:
            heapq.heappush(self.beam, (log_probability, new_state))
        else:
            heapq.heapreplace(self.beam, (log_probability, new_state))

    def get_largest(self, beam):
        return list(heapq.nlargest(1, beam))[0]

    def search(
        self,
        start_state: Dict,
        step_function: Callable[[Dict], Tuple[List[str], torch.Tensor, Tuple]],
        append_token_function: Callable[[Tuple, Tuple, str, float], Tuple]
    ) -> Tuple[List, float, Tuple]:
        state = BeamSearchState(state=start_state)
        heapq.heappush(self.beam, (0, state))

        done = False
        while not done:
            current_beam = self.beam

            self.beam = []
            found_non_terminal = False

            for (log_probability, state) in current_beam:
                if self.is_end_of_sequence(state.state, self.max_steps, state.sequence):
                    self.append(log_probability, state)

                else:
                    found_non_terminal = True
                    tokens, token_probabilities, extra_state = step_function(state.state)

                    tps = np.array(token_probabilities)
                    top_indices = heapq.nlargest(self.per_node_beam_size, range(len(tps)), tps.take)
                    top_tokens = [tokens[i] for i in top_indices]
                    top_probabilities = [token_probabilities[i] for i in top_indices]
                    sequence_length = len(state.sequence)
                    for i in range(len(top_tokens)):
                        if top_probabilities[i] > 1e-6:
                            #lp = ((log_probability * sequence_length) + math.log(top_probabilities[i])) / (sequence_length + 1.0)
                            lp = log_probability + math.log(top_probabilities[i])
                            t = top_tokens[i]
                            new_state = append_token_function(state.state, extra_state, t, lp)
                            new_sequence = state.sequence.copy()
                            new_sequence.append(t)

                            ns = BeamSearchState(sequence=new_sequence, log_probability=lp, state=new_state)

                            self.append(lp, ns)

            if not found_non_terminal:
                done = True

        output = self.get_largest(self.beam)[1]

        return output.sequence, output.log_probability, output.state, self.beam



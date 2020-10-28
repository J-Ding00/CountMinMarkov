
"""
Markov Chain Text Generator
"""

import numpy as np
from CountMin import CountMin

class Markov():
    def __init__(self,**kargs):
        self.start_tokens = []
        self.total_tokens = 0

    @staticmethod
    def pair_states(from_state, to_state):
        """Return representation of transition from from_state to to_state"""
        return from_state + (to_state[len(to_state)-1],)

    def state_prob(self, state):
        """Returns the proabability of being in a given state."""
        pass

    def trans_prob(self, from_state, to_state):
        """Returns the probability of transitioning from from_state to to_state."""
        pass

    def get_sentence(self, trans_bound):
        """
        Generate a sentence, using at most trans_bound predictions.
        """
        pass

    def update_model(self, from_state, to_state):
        """Update model based on observed transition from from_state to to_state."""
        pass

    def parse_text(self, path, tokenizer, encoding=None):
        """
        Parameters:
            path: A path to a .txt file to train the model on
            tokenizer: string => list of list of strings, in order
            encoding: Text encoding of the file at path
        """
        self.start_tokens = set(self.start_tokens)
        with open(path, 'r', encoding=encoding) as f:
            for line in f:
                tokens = tokenizer(line)
                self.total_tokens += len(tokens)
                for i in range(len(tokens) - 1):
                    self.start_tokens.add(tokens[i][0])
                    self.update_model(tokens[i], tokens[i+1])
        self.start_tokens = list(self.start_tokens)

class CountMinMarkov(Markov):
    def __init__(self, num_hash, length_table):
        self.sketch = CountMin(num_hash, length_table)
        super().__init__()

    def state_prob(self, state):
        """Returns the proabability of being in a given state."""
        return max(self.sketch.get_count(hash(state)) - self.sketch.expected_error(), 0) / self.total_tokens

    def trans_prob(self, from_state, to_state):
        """Returns the probability of transitioning from from_state to to_state."""
        return max(self.sketch.get_count(hash(self.pair_states(from_state, to_state))), 0) / max(self.sketch.get_count(hash(from_state)), 1)

    def get_sentence(self, start_state, trans_bound):
        """
        Generate text, from start_state, using at most trans_bound predictions.
        """
        sentence = ""
        for i in range(trans_bound):
            probs = np.array([self.trans_prob(start_state, start_state[1:]+(s,)) for s in self.start_tokens])
            avg = np.average(probs)
            std = np.std(probs)
            print(f"avg: {avg}, std {std}")
            probs /= np.sum(probs) # normalize probabilities
            next_word = np.random.choice(self.start_tokens, p=probs)
            sentence += " " + next_word
            start_state = start_state[1:]+(next_word,)
        return sentence
        

    def update_model(self, from_state, to_state):
        """
        Update model based on observed transition from from_state to to_state.
        States are indexable collections of strings.
        """
        self.sketch.update(hash(from_state))
        self.sketch.update(hash(self.pair_states(from_state, to_state)))

    @staticmethod
    def get_parameters(memory_bound):
        """
        Given a bound on the number of counters, return optimal
        parameters for the count-min sketch to minimize error.
        """
        pass


class ExactMarkov(Markov):
    pass

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
        pass

class CountMinMarkov(Markov):
    def __init__(self, num_hash, length_table, transition_count=3000):
        self.sketch = CountMin(num_hash, length_table, heavy_hitters=transition_count)
        super().__init__()

    def parse_text(self, path, tokenizer, num_hash, length_table, vocab_size=200, encoding=None):
        """
        Parameters:
            path: A path to a .txt file to train the model on
            tokenizer: string => list of list of strings, in order
            encoding: Text encoding of the file at path
        """
        # word_sketch = CountMin(num_hash, length_table, heavy_hitters=vocab_size)
        # with open(path, 'r', encoding=encoding) as f:
        #     for line in f:
        #         tokens = tokenizer(line)
        #         for i in range(len(tokens) - 1):
        #             word_sketch.update(tokens[i][0])
        # self.start_tokens = list(word_sketch.heavy_hitters.keys())
        #self.start_tokens = set(self.start_tokens)
        with open(path, 'r', encoding=encoding) as f:
            for line in f:
                tokens = tokenizer(line)
                for i in range(len(tokens) - 1):
                    #self.start_tokens.add(tokens[i][0])
                    #if all([word in word_sketch.heavy_hitters for word in tokens[i]]) and tokens[i+1][0] in word_sketch.heavy_hitters: 
                    self.total_tokens += 1
                    self.update_model(tokens[i], tokens[i+1])
        #self.start_tokens = list(self.start_tokens)


    def state_prob(self, state):
        """Returns the proabability of being in a given state."""
        return max(self.sketch.get_count(state, sub_error=False), 0) / self.total_tokens

    def trans_prob(self, from_state, to_state):
        """Returns the probability of transitioning from from_state to to_state."""
        #from_count = self.sketch.get_count(from_state, sub_error=False)
        #to_count = self.sketch.get_count(to_state, sub_error=False)
        return self.sketch.get_count(self.pair_states(from_state, to_state), sub_error=False)
        # pair_count = min(self.sketch.get_count(self.pair_states(from_state, to_state), sub_error=False), from_count, to_count)
        # return pair_count / from_count

    def get_start_state(self):
        pass

    def get_sentence(self, start_state, trans_bound):
        """
        Generate text, from start_state, using at most trans_bound predictions.
        """
        sentence = " ".join(start_state)
        probs = np.empty(len(self.sketch.heavy_hitters))
        hh_list = list(self.sketch.heavy_hitters.keys())
        for i in range(trans_bound):
            #probs = np.array([self.trans_prob(start_state, start_state[1:]+(s,)) for s in self.start_tokens], dtype=np.float32)
            for j in range(len(hh_list)):
                if hh_list[j][:-1] == start_state:
                    probs[j] = self.sketch.get_count(hh_list[j], sub_error=False)
                else:
                    probs[j] = 0
                j += 1
            avg = np.average(probs)
            std = np.std(probs)
            print(f"avg: {avg}, std {std}")
            prob_sum = np.sum(probs)
            if prob_sum == 0:
                return sentence
            probs /= prob_sum # normalize probabilities
            next_word = hh_list[np.random.choice(range(len(hh_list)), p=probs)]
            sentence += " " + next_word[-1]
            start_state = next_word[1:]
        return sentence
        

    def update_model(self, from_state, to_state):
        """
        Update model based on observed transition from from_state to to_state.
        States are indexable collections of strings.
        """
        #self.sketch.update(from_state, sub_error=False) # Needed for state prob, but not transition
        self.sketch.update(self.pair_states(from_state, to_state), sub_error=False)

    @staticmethod
    def get_parameters(memory_bound):
        """
        Given a bound on the number of counters, return optimal
        parameters for the count-min sketch to minimize error.
        """
        pass


class ExactMarkov(Markov):
    pass
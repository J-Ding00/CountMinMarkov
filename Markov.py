
"""
Markov Chain Text Generator
"""

import numpy as np
from collections import Counter
from nltk.util import ngrams
from nltk.tokenize import RegexpTokenizer 
from heapdict import heapdict
from collections import defaultdict
from CountMin import CountMin

class Markov():

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

    def get_sentence(self, start_state, trans_bound, stop_on_period=True):
        """
        Generate text, from start_state, using at most trans_bound predictions.
        If stop_on_period, finish text generation upon observing an ending period.
        """
        pass

    def update_model(self, from_state, to_state):
        """Update model based on observed transition from from_state to to_state."""
        pass

    @staticmethod
    def get_tokenizer(num_predictors):
        """
        Return a function to break a string into a list of
        num_predictors length shingles (tuples), preserving the order of
        these shingles. A typical value for num_predictors is 2 or 3.
        """
        def tokenizer(text):
            tokenizer = RegexpTokenizer("[a-zA-Z\-’'.]{1,}")
            tokens = tokenizer.tokenize(text)
            tokens = ngrams(tokens, num_predictors)
            return list(tokens)
        return tokenizer

    def parse_text(self, path, tokenizer=None, encoding=None):
        """
        Parameters:
            path: A path to a .txt file to train the model on
            tokenizer: string => list of tuple of strings, in order
            encoding: Text encoding of the file at path
        """
        if tokenizer is None:
            tokenizer = self.get_tokenizer(3)
        with open(path, 'r', encoding=encoding) as f:
            tokens = tokenizer(f.read())
            for i in range(len(tokens) - 1):
                self.total_tokens += 1
                self.update_model(tokens[i], tokens[i+1])

class MaxDegreeMarkov(Markov):
    """
    Markov-chain text generator seeking to maximize the number of distinct sentences
    which can be generated using only max_trans transitions.
    """
    def __init__(self, max_trans):
        self.total_tokens = 0
        self.out_trans = {} # Map state to set of transitions with that prefix
        self.in_trans = {} # Map state to set of transitions ending with that suffix
        self.max_trans = max_trans
        self.trans_pq = heapdict() # Elements are transitions with priority in_degree(from_state)*out_degree(to_state)

    def state_prob(self, state):
        """Returns the proabability of being in a given state."""
        if state in self.out_trans:
            return len(self.out_trans[state]) / len(self.trans_pq)
        else:
            return 0

    def trans_prob(self, from_state, to_state):
        """Returns the probability of transitioning from from_state to to_state."""
        full_trans = self.pair_states(from_state,to_state)
        if from_state in self.out_trans and full_trans in self.out_trans[from_state]:
            return 1 / len(self.out_trans[from_state])
        else:
            return 0

    def get_start_state(self):
        """
        Returns a random starting state, which begins with a capital letter. 
        This is done by uniformly choosing a transition then extracting the from_state.
        """
        start_states = []
        for trans in self.trans_pq:
            if str.isupper(trans[0][0]):
                start_states.append(trans)
        return start_states[np.random.randint(len(start_states))][:-1]

    def get_sentence(self, start_state=None, trans_bound=30, stop_on_period=True):
        """
        Generate text, from start_state, using at most trans_bound predictions.
        If stop_on_period, finish text generation upon observing an ending period.
        """
        if start_state is None:
            start_state = self.get_start_state()
        sentence = " ".join(start_state)
        curr_state = start_state
        for _ in range(trans_bound):
            if curr_state not in self.out_trans:
                return sentence + '.'
            trans = list(self.out_trans[curr_state])[np.random.randint(len(self.out_trans[curr_state]))]
            sentence += ' ' + trans[-1]
            if stop_on_period and sentence[-1] == '.': # Ending on a period, end here
                return sentence
            curr_state = trans[1:]
        return sentence + '.'

    def parse_text(self, path, tokenizer=None, encoding=None):
        """
        Parameters:
            path: A path to a .txt file to train the model on
            tokenizer: string => list of tuple of strings, in order
            encoding: Text encoding of the file at path
        """
        if tokenizer is None:
            tokenizer = self.get_tokenizer(3)
        with open(path, 'r', encoding=encoding) as f:
            text = f.read().split('.')
            np.random.shuffle(text)
            tokens = tokenizer(". ".join(text))
            for i in range(len(tokens) - 1):
                self.total_tokens += 1
                self.update_model(tokens[i], tokens[i+1])

    def prediction_space(self, num_trans):
        """
        Returns the number of sentences which can be generated at each token length up to num_trans.
        """
        num_sentences = [len(self.out_trans)]
        sentences_at_state = {}
        for state in self.out_trans:
            sentences_at_state[state] = 1
        for _ in range(num_trans):
            sentence_count = 0
            curr_sentences_at_state = defaultdict(int)
            for state in sentences_at_state:
                if state in self.out_trans:
                    for trans in self.out_trans[state]:
                        to_state = trans[1:]
                        curr_sentences_at_state[to_state] += sentences_at_state[state]
                        sentence_count += sentences_at_state[state]
            sentences_at_state = curr_sentences_at_state
            num_sentences.append(sentence_count)
        return num_sentences

    def get_priority(self, from_state, to_state):
        """
        Return product of in-degree of from_state with out-degree of to_state.
        """
        if from_state in self.in_trans and to_state in self.out_trans:
            return len(self.in_trans[from_state]) * len(self.out_trans[to_state])
        else:
            return 0

    def update_model(self, from_state, to_state):
        """
        Update model based on observed transition from from_state to to_state.
        States are indexable collections of strings.
        """
        full_trans = self.pair_states(from_state, to_state)
        if from_state in self.out_trans:
            self.out_trans[from_state].add(full_trans)
        else:
            self.out_trans[from_state] = {full_trans}
        if to_state in self.in_trans:
            self.in_trans[to_state].add(full_trans)
        else:
            self.in_trans[to_state] = {full_trans}
        priority = self.get_priority(from_state,to_state)
        self.trans_pq[full_trans] = priority
        while len(self.trans_pq) > self.max_trans:
            # Must remove some transitions to keep memory usage reasonable
            trans, priority = self.trans_pq.popitem()
            from_state = trans[:-1]
            to_state = trans[1:]
            curr_priority = self.get_priority(from_state,to_state)
            if priority < curr_priority: # Priority increased at some point
                self.trans_pq[trans] = curr_priority
            else:
                # Remove low priority transition
                self.in_trans[to_state].remove(trans)
                self.out_trans[from_state].remove(trans)
                if len(self.in_trans[to_state]) == 0:
                    self.in_trans.pop(to_state)
                if len(self.out_trans[from_state]) == 0:
                    self.out_trans.pop(from_state)

    def expected_trans(self, trans_bound=30):
        """
        Returns the expected number of word predictions which can occur until
        there are no outgoing transitions from the current state, bounded by trans_bound transitions.
        Assumes sentence generation continues after ending on a period.
        """
        expected = 0
        trans_weights = {}
        init_weight = 1 / len(self.trans_pq)
        for trans in self.trans_pq:
            trans_weights[trans] = init_weight # Total weights should sum to 1
        for trans_num in range(1,trans_bound+1):
            iter_weight = 0
            curr_trans_weights = defaultdict(float)
            for trans in trans_weights:
                to_state = trans[1:]
                if to_state in self.out_trans:
                    split_weight = trans_weights[trans] / len(self.out_trans[to_state])
                    for to_trans in self.out_trans[to_state]:
                        curr_trans_weights[to_trans] += split_weight
                else:
                    iter_weight += trans_weights[trans]
            expected += iter_weight * trans_num
            trans_weights = curr_trans_weights
        return expected + (sum(trans_weights.values()) * (trans_bound + 1))

class BatchMarkov(MaxDegreeMarkov):
    """
    Markov-chain text generator seeking to maximize the expected value of the
    number of transitions traversed in the model before reaching a state with
    no outgoing transitions.

    Only including transitions which form a cycle would accomplish this, but
    max_trans transitions are stored even if having fewer transitions would
    increase the expected number of transitions before a state without outgoing transitions.
    """
    def __init__(self, max_trans, batch_size=1000, smoothing=0.1):
        super().__init__(max_trans)
        self.smoothing = smoothing
        self.batch_size = batch_size
        self.default_priority = 1

    def update_model(self, from_state, to_state):
        """
        Update model based on observed transition from from_state to to_state.
        States are indexable collections of strings.
        """
        full_trans = self.pair_states(from_state, to_state)
        if from_state in self.out_trans:
            self.out_trans[from_state].add(full_trans)
        else:
            self.out_trans[from_state] = {full_trans}
        if full_trans not in self.trans_pq:
            self.trans_pq[full_trans] = self.default_priority
        if len(self.trans_pq) > self.max_trans + self.batch_size:
            # Apply inference, accumulating probabilities from transitions with to_state same as from_state in trans
            next_priorities = defaultdict(float)
            for trans in self.trans_pq:
                to_state = trans[1:]
                if to_state in self.out_trans: # Otherwise, we have no next transition available
                    split_priority = self.trans_pq[trans] / len(self.out_trans[to_state])
                    for to_trans in self.out_trans[to_state]:
                        next_priorities[to_trans] += split_priority
            # Add smoothing
            magnitude = np.linalg.norm(list(next_priorities.values()))
            smoothing_factor = (1 - self.smoothing) / magnitude if magnitude > 0 else 1
            smoothing_constant = (self.smoothing / len(next_priorities)) / magnitude if magnitude > 0 else 1
            self.trans_pq = heapdict()
            for trans in next_priorities:
                self.trans_pq[trans] = next_priorities[trans] * smoothing_factor + smoothing_constant
            while len(self.trans_pq) > self.max_trans:
                # Must remove some transitions to keep memory usage reasonable
                trans, priority = self.trans_pq.popitem()
                from_state = trans[:-1]
                to_state = trans[1:]
                # Remove low priority transition
                self.out_trans[from_state].remove(trans)
                if len(self.out_trans[from_state]) == 0:
                    self.out_trans.pop(from_state)
            # Set default priority to mean of priorities
            self.default_priority = sum(p for p in self.trans_pq.values()) / len(self.trans_pq)
            

class CountMinMarkov(Markov):
    def __init__(self, num_hash, length_table, transition_count=None, track_state_probs=False, sub_error=False):
        """
        An approximate Markov-Chain text generator.
        Parameters:
            num_hash: The number of hash functions (and tables) used in the backing count-min sketch(s)
            length_table: The number of elements in each table for the backing count-min sketch(s)
            transition_count: The number of heavy hitting (most frequent) transitions stored for generation
            track_state_probs: Track counts of each state, state_prob will be defined iff track_state_probs
            sub_error: Subtract the expected value of the error on each count when using the count-min sketch
        """
        if transition_count is None:
            transition_count = num_hash * length_table
        self.trans_sketch = CountMin(num_hash, length_table, heavy_hitters=transition_count)
        self.track_state_probs = track_state_probs
        self.sub_error = sub_error
        self.total_tokens = 0
        if track_state_probs:
            self.state_sketch = CountMin(num_hash, length_table)

    def state_prob(self, state, sub_error=False):
        """Returns the proabability of being in a given state."""
        return self.state_sketch.get_count(state, sub_error=sub_error) / self.total_tokens

    def trans_prob(self, from_state, to_state, sub_error=False):
        """Returns the probability of transitioning from from_state to to_state."""
        from_count = self.state_sketch.get_count(from_state, sub_error=sub_error)
        to_count = self.state_sketch.get_count(to_state, sub_error=sub_error)
        pair_count = min(self.trans_sketch.get_count(self.pair_states(from_state, to_state), sub_error=sub_error), from_count, to_count)
        return pair_count / max(from_count, 1)

    def get_start_state(self):
        """
        Returns a random starting state, which begins with a capital letter. 
        """
        probs = np.empty(len(self.trans_sketch.heavy_hitters))
        hh_list = list(self.trans_sketch.heavy_hitters.keys())
        for i in range(len(hh_list)):
            if str.isupper(hh_list[i][0][0]): # Limit to text starting with capital letter
                probs[i] = self.trans_sketch.get_count(hh_list[i], sub_error=self.sub_error)
            else:
                probs[i] = 0
        probs /= np.sum(probs)
        return hh_list[np.random.choice(range(len(hh_list)), p=probs)][:-1]

    def get_sentence(self, start_state=None, trans_bound=30, stop_on_period=True, sub_error=None):
        """
        Generate text, from start_state, using at most trans_bound predictions.
        If stop_on_period, finish text generation upon observing an ending period.
        """
        if start_state is None:
            start_state = self.get_start_state()
        if sub_error is None:
            sub_error = self.sub_error
        sentence = " ".join(start_state)
        probs = np.empty(len(self.trans_sketch.heavy_hitters))
        hh_list = list(self.trans_sketch.heavy_hitters.keys())
        for i in range(trans_bound):
            for j in range(len(hh_list)):
                if hh_list[j][:-1] == start_state:
                    probs[j] = self.trans_sketch.get_count(hh_list[j], sub_error=sub_error)
                else:
                    probs[j] = 0
                j += 1
            prob_sum = np.sum(probs)
            if prob_sum == 0:
                return sentence
            probs /= prob_sum # normalize probabilities
            next_trans = hh_list[np.random.choice(range(len(hh_list)), p=probs)]
            sentence += " " + next_trans[-1]
            if stop_on_period and sentence[-1] == '.': # Ending on a period, end here
                return sentence
            start_state = next_trans[1:]
        return sentence + '.'
        

    def update_model(self, from_state, to_state):
        """
        Update model based on observed transition from from_state to to_state.
        States are indexable collections of strings.
        """
        if self.track_state_probs:
            self.state_sketch.update(from_state, sub_error=self.sub_error)
        self.trans_sketch.update(self.pair_states(from_state, to_state), sub_error=self.sub_error)

class ExactMarkov(Markov):
    def __init__(self, track_state_probs=False):
        """
        A Markov-chain text generator.
        Parameters:
            track_state_probs: Track counts of each state, state_prob will be defined iff track_state_probs
        """
        self.bag_of_states = Counter()
        self.bag_of_transitions = Counter()
        self.total_tokens = 0
        self.track_state_probs = track_state_probs

    def state_prob(self, state):
        """Returns the proabability of being in a given state."""
        return self.bag_of_states[state] / self.total_tokens

    def trans_prob(self, from_state, to_state):
        """Returns the probability of transitioning from from_state to to_state."""
        return self.bag_of_transitions[self.pair_states(from_state, to_state)] / self.bag_of_states[from_state]

    def get_start_state(self):
        """
        Returns a random starting state, which begins with a capital letter. 
        """
        probs = np.empty(len(self.bag_of_transitions))
        trans_list = list(self.bag_of_transitions.keys())
        for i in range(len(trans_list)):
            if str.isupper(trans_list[i][0][0]): # Limit to text starting with capital letter
                probs[i] = self.bag_of_transitions[trans_list[i]]
            else:
                probs[i] = 0
        probs /= np.sum(probs)
        return trans_list[np.random.choice(range(len(trans_list)), p=probs)][:-1]

    def get_sentence(self, start_state=None, trans_bound=30, stop_on_period=True):
        """
        Generate a sentence, using at most trans_bound predictions.
        If stop_on_period, finish text generation upon observing an ending period.
        """
        if start_state is None:
            start_state = self.get_start_state()
        sentence = " ".join(start_state)
        probs = np.empty(len(self.bag_of_transitions))
        trans_list = list(self.bag_of_transitions.keys())
        for i in range(trans_bound):
            for j in range(len(trans_list)):
                if trans_list[j][:-1] == start_state:
                    probs[j] = self.bag_of_transitions[trans_list[j]]
                else:
                    probs[j] = 0
                j += 1
            prob_sum = np.sum(probs)
            if prob_sum == 0:
                return sentence
            probs /= prob_sum # normalize probabilities
            next_trans = trans_list[np.random.choice(range(len(trans_list)), p=probs)]
            sentence += " " + next_trans[-1]
            if stop_on_period and sentence[-1] == '.': # Ending on a period, end here
                return sentence
            start_state = next_trans[1:]
        return sentence + '.'

    def update_model(self, from_state, to_state):
        """Update model based on observed transition from from_state to to_state."""
        if self.track_state_probs:
            self.bag_of_states[from_state] += 1
        self.bag_of_transitions[self.pair_states(from_state, to_state)] += 1

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

    def get_sentence(self, start_state, trans_bound):
        """
        Generate text, from start_state, using at most trans_bound predictions.
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
    def __init__(self, max_trans, iterations):
        self.total_tokens = 0
        self.out_trans = {} # Map state to set of transitions with that prefix
        self.in_trans = {} # Map state to set of transitions ending with that suffix
        self.max_trans = max_trans
        self.trans_pq = heapdict() # Elements are transitions with priority in_degree(from_state)*out_degree(to_state)

    def get_start_state(self):
        """
        Returns a random starting state, which begins with a capital letter. 
        """
        start_states = []
        for state in self.out_trans:
            if str.isupper(state[0][0]):
                start_states.append(state)
        return start_states[np.random.randint(len(start_states))]

    def get_sentence(self, start_state=None, trans_bound=30):
        """
        Generate text, from start_state, using at most trans_bound predictions.
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
            if sentence[-1] == '.': # Ending on a period, end here
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
        if from_state in self.in_trans and to_state in self.out_trans:
            return len(self.in_trans[from_state]) * len(self.out_trans[to_state])
        else:
            return 0

    def update_model(self, from_state, to_state):
        """
        Update model based on observed transition from from_state to to_state.
        States are indexable collections of strings.
        """
        #state = from_state[1:]
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

    def get_sentence(self, start_state=None, trans_bound=30, sub_error=None):
        """
        Generate text, from start_state, using at most trans_bound predictions.
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
            if sentence[-1] == '.': # Ending on a period, end here
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

    def get_sentence(self, start_state=None, trans_bound=30):
        """
        Generate a sentence, using at most trans_bound predictions.
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
            if sentence[-1] == '.': # Ending on a period, end here
                return sentence
            start_state = next_trans[1:]
        return sentence + '.'

    def update_model(self, from_state, to_state):
        """Update model based on observed transition from from_state to to_state."""
        if self.track_state_probs:
            self.bag_of_states[from_state] += 1
        self.bag_of_transitions[self.pair_states(from_state, to_state)] += 1
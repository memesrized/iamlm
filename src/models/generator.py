import json
from copy import deepcopy

import numpy as np
from compress_fasttext.models import CompressedFastTextKeyedVectors
from numpy.random import randint
from src import PROJECT_PATHS


class Generator:
    def __init__(self, embeddings):
        self.embeddings = CompressedFastTextKeyedVectors.load(str(embeddings))
        with open(PROJECT_PATHS.configs / "filthy_words.json") as file:
            self.filthy_words = [self._get_embeddings(x) for x in json.load(file)]

    def _get_next_n_gram(self, seq, dict_, context=None, beam_size=1):
        # TODO: similarity based on seq itself
        if not dict_.get(seq):
            return " ".join(seq), None
        tokens, nums = zip(*dict_[seq].items())
        prob = np.array(nums) / sum(nums)
        if context:
            # TODO: make more flexible weights
            beam = np.random.choice(tokens, size=beam_size, p=np.array(prob))
            new_token = self._most_similar(context, beam)
        else:
            new_token = np.random.choice(tokens, p=np.array(prob))
        return seq[0], seq[1:] + (new_token,)

    def make_sequence(
        self, start_token, dict_, end_token, context=None, beam_size=1, len_=30
    ):
        # for np.random.choice array should be 1-d
        start_n_gram = [x for x in dict_ if start_token == x[0]]
        prev_word, next_n_gram = self._get_next_n_gram(
            start_n_gram[randint(0, len(start_n_gram))], dict_
        )
        tokens = [prev_word]
        if next_n_gram:
            i = 0
            while next_n_gram and (i < len_):
                i += 1
                prev_word, next_n_gram = self._get_next_n_gram(
                    next_n_gram, dict_, context, beam_size
                )
                tokens.append(prev_word)
            if next_n_gram:
                # to be sure that last token was included
                # if len_ was reached
                tokens.append(" ".join(next_n_gram))
        return " ".join([x for x in tokens if x])

    def _most_similar(self, context, tokens):
        # TODO: compare only precomputed context vector to improve speed
        sims = [self.embeddings.n_similarity(context, [x]) for x in tokens]
        return tokens[np.argmax(sims)]

    def beam_search(
        self, start_token, dict_, end_token, context=None, beam_size=3, len_=30
    ):
        start_n_gram = [x for x in dict_ if start_token == x[0]]
        beam_data = [
            {
                "tokens": [],
                "ngram": start_n_gram[randint(0, len(start_n_gram))],
                "log_prob": 0,
            }
            for x in range(beam_size)
        ]
        if not context:
            context = self._calculate_mean_vec(self.filthy_words) * (-1)
        else:
            context = self._calculate_mean_vec(context)

        result = []
        i = 0
        while beam_data and i < len_:
            i += 1
            ended_step, beam_data = self._iter(beam_data, dict_, end_token, context)
            if ended_step:
                result.extend(ended_step)

        if beam_data:
            result.extend(beam_data)

        for res in result:
            res["tokens"].extend(res["ngram"])
        # len - 1 because we don't want to take into account start_token
        log_probs = np.array([x["log_prob"] / (len(x["tokens"]) - 1) for x in result])
        i = np.argmin(log_probs)
        return " ".join(result[i]["tokens"]), result

    def _iter(self, list_data, dict_, end_token, context):
        # TODO: consider only top-k for every possible token
        beam_size = len(list_data)

        # create dict of all possible tokens
        step_dict = {}
        for i, message in enumerate(list_data):
            message_ngram_dict = dict_[message["ngram"]]
            temp_dict = {
                (i, message["ngram"], key): value
                for key, value in message_ngram_dict.items()
            }
            step_dict.update(temp_dict)

        # TODO: rename tokens
        # find top-beam_size
        tokens, nums = zip(*step_dict.items())

        if context is None:
            prob = np.array(nums) / sum(nums)
            new_token_idx = np.random.choice(
                range(len(tokens)), size=beam_size, p=np.array(prob)
            )
        else:
            scores = np.array([self._cosine_score(x[-1], context) for x in tokens])
            prob = scores / sum(scores)
            new_token_idx = np.random.choice(
                range(len(tokens)), size=beam_size, p=np.array(prob)
            )

        # create output
        new_list_data = []
        ended = []
        for i in new_token_idx:
            current = deepcopy(list_data[tokens[i][0]])

            new_ngram = current["ngram"][1:] + (tokens[i][-1],)
            old_token = current["ngram"][0]
            current["tokens"].append(old_token)

            temp_record_dict = {
                "tokens": current["tokens"],
                "ngram": new_ngram,
                "log_prob": current["log_prob"] - np.log(prob[i]),
            }
            if tokens[i][-1] == end_token:
                ended.append(temp_record_dict)
            else:
                new_list_data.append(temp_record_dict)

        return (ended, new_list_data)

    def _calculate_sim(self, a, b):
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        # TODO: check for zeros at norm_a and norm_b
        # instead of product for nan
        product = dot_product / (norm_a * norm_b)

        # we don't want to skip all oov words
        # so let's just give them low score
        return product if not np.isnan(product) and product > 0 else 0.1

    def _calculate_mean_vec(self, tokens):
        embs = [self.embeddings[self._normalize_string(x)] for x in tokens]
        return sum(embs) / len(tokens)

    def _cosine_score(self, word, context_vector):
        cosine_sim = self._calculate_sim(
            self.embeddings[self._normalize_string(word)], context_vector
        )
        return cosine_sim

    def _normalize_string(self, text):
        return text.replace("ё", "е")

    def _get_embeddings(self, word):
        return self.embeddings[self._normalize_string(word)]

    def _filthy_words_distance(self, word):
        distances = []
        word_emb = self.embeddings[self._normalize_string(word)]
        for word in self.filthy_words:
            distances.append(self._calculate_sim(word_emb, ))

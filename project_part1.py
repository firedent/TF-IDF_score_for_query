# Import Libraries and Modules here...
import spacy
from math import log
from collections import defaultdict, Counter
from itertools import combinations, repeat
import copy


class InvertedIndex:
    def __init__(self):
        # You should use these variable to store the term frequencies for tokens and entities...
        self.tf_tokens = defaultdict(lambda: defaultdict(float))
        self.tf_entities = defaultdict(lambda: defaultdict(float))

        # You should use these variable to store the inverse document frequencies for tokens and entities...
        self.idf_tokens = defaultdict(float)
        self.idf_entities = defaultdict(float)

        self.ql = list()
        self._total_docs = 0

    # Your implementation for indexing the documents...
    def index_documents(self, documents):
        dict_entities = self.tf_entities
        dict_tokens = self.tf_tokens
        self._total_docs = len(documents)
        nlp = spacy.load("en_core_web_sm")

        def could_be_token(t):
            if t.is_stop or t.is_punct:
                return False
            else:
                return True

        # e for entity, d for document ID
        def count_entity(e, d):
            dict_entities[e.text][d] += 1

        # t for token, d for document ID
        def count_token(t, d):
            if could_be_token(t):
                dict_tokens[t.text][d] += 1

        for doc_id_text in documents.items():
            tokens = nlp(doc_id_text[1])
            entities = tokens.ents
            for ent in entities:
                # count this entity
                count_entity(ent, doc_id_text[0])
                # if this entity is multi-entity, count those tokens contained in it
                if len(ent) >= 2:
                    for token in ent:
                        count_token(token, doc_id_text[0])
            for token in tokens:
                # if this token in an entity, it already be count in pass.
                if token.ent_iob == 1 or token.ent_iob == 3:
                    continue
                count_token(token, doc_id_text[0])

        for tokenText, tokenCounts in self.tf_tokens.items():
            # total docs containing token
            tdct = len(tokenCounts)
            self.idf_tokens[tokenText] = 1.0 + log(self._total_docs / (1.0 + tdct))

        for entityText, entityConuts in self.tf_entities.items():
            # total docs containing entity
            tdce = len(entityConuts)
            self.idf_entities[entityText] = 1.0 + log(self._total_docs / (1.0 + tdce))

    # Your implementation to split the query to tokens and entities...
    def split_query(self, Q, DoE):
        query_string_list = Q.split()
        # convert entity included in Doe into tuple splitted by SPACE
        DoE_split_set = set([tuple(i.split()) for i in DoE.keys()])
        # the max length of entity in DoE
        DoE_max_length = max(map(len, DoE_split_set))
        e_and_k = dict()

        # get free keywords list
        # e for tuple of splitted entity, q for list of splitted query
        def get_free_keywords(entity_list, q):
            entity_counter = Counter()
            query_counter = Counter(q)
            for e in entity_list:
                entity_counter.update(e)
            if len(entity_counter - query_counter) == 0:
                query_counter.subtract(entity_counter)
                query_counter = +query_counter
                r = list()
                for token_text, time in query_counter.items():
                    for _ in range(time):
                        r.append(token_text)
                return True, r
            else:
                return False, []

        probable_entities_set = set()
        for i in range(1, DoE_max_length + 1):
            probable_entities_set.update(combinations(Q.split(), i))

        probable_entities_set &= DoE_split_set
        # remove the entity whose tokens not include in query
        # It may not affect final result without sorting DoE list

        k = 0
        for i in range(len(probable_entities_set) + 1):
            for entities in combinations(probable_entities_set, i):
                r = get_free_keywords(entities, query_string_list)
                if r[0]:
                    e_and_k[k] = dict(tokens=r[1], entities=list(map(lambda x: ' '.join(x), entities)))
                    k += 1
        return e_and_k

    # Your implementation to return the max score among all the query splits...
    def max_score_query(self, query_splits, doc_id):
        max_score = 0
        max_result = dict()
        for key, split in query_splits.items():
            s_entity = sum(
                [
                    (1.0 + log(self.tf_entities[entity][doc_id])) * self.idf_entities[entity]
                    if self.tf_entities[entity][doc_id] != 0
                    else 0
                    for entity in split['entities']
                ]
            )

            s_token = sum(
                [
                    (1.0 + log(1.0 + log(self.tf_tokens[token][doc_id]))) * self.idf_tokens[token]
                    if self.tf_tokens[token][doc_id] != 0
                    else 0
                    for token in split['tokens']
                ]
            )
            score = s_entity + 0.4 * s_token
            result = dict(tokens=split['tokens'], entities=split['entities'])
            self.ql.append(dict(tokens=split['tokens'], entities=split['entities'],
                                s_token=s_token, s_entity=s_entity, s_combined=score))
            if score >= max_score:
                max_score = score
                max_result = result

        return max_score, max_result
        # Output should be a tuple (max_score, {'tokens': [...], 'entities': [...]})

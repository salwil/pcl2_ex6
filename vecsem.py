
#!/usr/bin/env python3
#-*- coding: utf-8 -*-
# Roger Rüttimann rroger 02-914-471
# Salome Wildermuth salomew 10-289-544


from collections import defaultdict
from typing import Dict, Iterable, List, TextIO, Tuple
import numpy as np
import gzip
from spacy.lemmatizer import Lemmatizer
from spacy.lang.en import LEMMA_INDEX, LEMMA_EXC, LEMMA_RULES
from numpy import dot
from numpy.linalg import norm


class HomemadeDSM():
  """Implements a WordEmbedding Exercise."""

  def __init__(self, train_corpus: TextIO):
    """Initialize and train the model.

    Args:
        train_corpus: A open file or text stream with annoted text for trainiing of the model.

    """

    lemmatizer = Lemmatizer(LEMMA_INDEX, LEMMA_EXC, LEMMA_RULES)
    self.coocurrenz = defaultdict(lambda: defaultdict(int))
    self.all_verbs = set()
    self.all_nouns = set()
    for line in train_corpus:
      verbs = set()
      nouns = set()
      for word_pos in line.split():
        (word, pos) = str(word_pos).split('_')
        pos_type = pos[0]
        if (pos_type != 'V') and (pos_type != 'N'):
          continue # nothing to do with the word
        word = word[2:]
        if pos_type == 'V':
          verb = lemmatizer(word, u'VERB')[0]
          verbs.add(verb)
        elif pos_type == 'N':
          noun = lemmatizer(word, u'NOUN')[0]
          nouns.add(noun)
      self.all_nouns.update(nouns)
      self.all_verbs.update(verbs)
      for noun in nouns:
        for verb in verbs:
          self.coocurrenz[noun][verb] += 1
    self.all_nouns = sorted(self.all_nouns)
    self.all_verbs = sorted(self.all_verbs)

  def get_vec(self, noun: str) -> np.array:
    """returns word embedding for noun."""
    
    noun = noun.lower()
    vec = []
    for verb in self.all_verbs:
      vec.append(self.coocurrenz[noun][verb])
    return np.array(vec)

  def get_similarity(self, noun1: str, noun2: str) -> float:
    """Calculates cosinus similarity of two nouns."""

    a = self.get_vec(noun1)
    b = self.get_vec(noun2)
    return dot(a, b)/(norm(a)*norm(b))

  def get_most_similar(self, ref_noun: str, comp_nouns: List[str]) -> Tuple[float, str]:
    """returns tuple of similarity value and noun of most similar noun in comp_nouns list compared to ref_noun."""

    sim_list = [(self.get_similarity(ref_noun, noun), noun) for noun in comp_nouns]
    return max(sim_list, key= lambda x: x[0])

if __name__ == '__main__':
  with gzip.open('a2/CLOB_pos_tagged.txt.gz', mode='r') as reader:
    hdsm = HomemadeDSM(train_corpus = reader)
    print(hdsm.get_vec('machiavellian'))
    print(hdsm.get_vec('Israel').shape)
    print(hdsm.get_vec('Switzerland').shape)
    print(hdsm.get_vec('Roger').shape)
    print('Switzerland' in hdsm.all_nouns)
    print(hdsm.get_similarity('Switzerland','Israel'))
    print(hdsm.get_similarity('Switzerland', 'Gaza'))
    print(hdsm.get_similarity('Israel', 'Gaza'))
    print(hdsm.get_most_similar('Gaza', ['Switzerland', 'Israel', 'Thailand']))


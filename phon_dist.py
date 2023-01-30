## Licensed under the Apache License, Version 2.0 (the "License");
## you may not use this file except in compliance with the License.
## You may obtain a copy of the License at
##
##      http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software
## distributed under the License is distributed on an "AS IS" BASIS,
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
## See the License for the specific language governing permissions and
## limitations under the License.
##
## Author: Richard Sproat (rws@xoba.com)
"""Computes phonetic distance.
"""

import collections
import numpy as np
import random

SYL_DIVIDER = "."


class PhoneticDistance:
  """
  """
  def __init__(self, phonetic_definitions, fixed_dim=None):
    self._read_phonetic_file(phonetic_definitions)
    offset = 0
    self._exact_alliteration_group = {}
    for c in self._consonants:
      self._exact_alliteration_group[c] = offset
      offset += 1
    self._rough_alliteration_group = {}
    seen_consonants = set()
    for equivalence in self._onset_equivalences:
      for c in equivalence:
        self._rough_alliteration_group[c] = offset
        seen_consonants.add(c)
      offset += 1
    # If a consonant is not accounted for it is its own equivalence class.
    for c in self._consonants:
      if c not in seen_consonants:
        self._rough_alliteration_group[c] = offset
        offset += 1
    self._exact_rhyme_group = {}
    for v in self._vowels:
      for c in self._consonants:
        self._exact_rhyme_group[v, c] = offset
        offset += 1
    self._rough_rhyme_group = {}
    seen_consonants = set()
    for v in self._vowels:
      for equivalence in self._coda_equivalences:
        for c in equivalence:
          self._rough_rhyme_group[v, c] = offset
          seen_consonants.add(c)
        offset += 1
    for c in self._consonants:
      if c not in seen_consonants:
        for v in self._vowels:
          self._rough_rhyme_group[v, c] = offset
          offset += 1
    self._vector_size = offset
    if fixed_dim:
      f"Error: fixed_dim={fixed_dim} < vector_size={self._vector_size}"
      assert fixed_dim >= self._vector_size, msg
      self._vector_size = fixed_dim
    self._zero_vector = np.array([0.0] * self._vector_size)
    self._all_syls = set()
    for c1 in self._consonants:
      for v in self._vowels:
        for c2 in self._consonants:
          self._all_syls.add(f"{c1}{v}{c2}".replace("-", ""))
    self._cache = {}
      
  def clear_cache(self): self._cache.clear()

  def _read_phonetic_file(self, phonetic_definitions):
    self._zero = "-"
    self._consonants = set([self._zero])
    self._vowels = set()
    self._onset_equivalences = []
    self._coda_equivalences = []
    with open(phonetic_definitions) as s:
      for line in s:
        try:
          tag, rest = line.strip().split("\t")
        except ValueError:
          if len(line.strip()) > 0:
            print(f"Skipping line: {line.strip()}")
          continue
        rest = rest.split()
        if tag == "C":
          for r in rest:
            self._consonants.add(r)
        elif tag == "V":
          for r in rest:
            self._vowels.add(r)
        elif tag == "Onset":
          equivalence = []
          for r in rest:
            equivalence.append(r)
          self._onset_equivalences.append(equivalence)
        elif tag == "Coda":
          equivalence = []
          for r in rest:
            equivalence.append(r)
          self._coda_equivalences.append(equivalence)
        else:
          print(f"Skipping line: {line.strip()}")

  def _parse_cvc_syllable(self, syllable):
    syllable = list(syllable)
    if not syllable[0] in self._consonants:
      syllable = [self._zero] + syllable
    if not syllable[-1] in self._consonants:
      syllable = syllable + [self._zero]
    try:
      onset, vowel, coda = syllable
      return onset, (vowel, coda)
    except ValueError:
      return None, None

  @property
  def vector_size(self): return self._vector_size

  def build_vector(self, syllable):
    vector = self._zero_vector.copy()
    onset, rhyme = self._parse_cvc_syllable(syllable)
    vector[self._exact_alliteration_group[onset]] = 1
    vector[self._rough_alliteration_group[onset]] = 1
    # Differences in rhymes count for more than differences in onsets.
    vector[self._exact_rhyme_group[rhyme]] = 2
    vector[self._rough_rhyme_group[rhyme]] = 2
    return vector / np.linalg.norm(vector)
  
  def phonetic_distance(self, s1, s2):
    if (s1, s2) in self._cache:
      return self._cache[s1, s2]
    v1 = self.build_vector(s1)
    v2 = self.build_vector(s2)
    dist = np.linalg.norm(v1 - v2)
    self._cache[s1, s2] = dist
    return dist
  
  def closest_syllables(self, s, lim=1.5):
    rank = []
    for s2 in self._all_syls:
      dist = self.phonetic_distance(s, s2)
      if dist <= lim:
        rank.append((dist, s, s2))
    rank.sort()
    return rank

  def randgen(self):
    syl = [random.choice(list(self._consonants)),
           random.choice(list(self._vowels)),
           random.choice(list(self._consonants))]
    return "".join(syl).replace("-", "")

  @property
  def zero_vector(self): return self._zero_vector
  

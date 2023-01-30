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
"""Computes distances between embeddings.
"""

import collections
import concepts
import numpy as np

from absl import app
from absl import flags

flags.DEFINE_string("embeddings", "emb.tsv", "Embeddings file")

FLAGS = flags.FLAGS


ALL_EMBEDDINGS = {}
ALL_EMBEDDINGS_LOADED = False


def _load_embeddings(path):
  global ALL_EMBEDDINGS_LOADED
  primary_embeddings = {}
  secondary_embeddings = {}
  with open(path) as s:
    for line in s:
      c, e = line.strip("\n").split("\t")
      e = np.array(eval(e))
      if c in concepts.CONCEPTS:
        primary_embeddings[c] = e
      else:
        secondary_embeddings[c] = e
      ALL_EMBEDDINGS[c] = e
  ALL_EMBEDDINGS_LOADED = True
  return primary_embeddings, secondary_embeddings


def embedding_distance(e1, e2):
  try:
    return np.linalg.norm(e1 - e2)
  except TypeError:
    return np.linalg.norm(np.array(e1) - np.array(e2))


def concept_distance(c1, c2):
  assert ALL_EMBEDDINGS_LOADED
  return embedding_distance(ALL_EMBEDDINGS[c1], ALL_EMBEDDINGS[c2])


def find_closest_embeddings(path):
  primary_embeddings, secondary_embeddings = _load_embeddings(path)
  result = collections.defaultdict(str)
  for e2 in secondary_embeddings:
    best_dist = 1_000_000
    best_e1 = None
    for e1 in primary_embeddings:
      dist = embedding_distance(primary_embeddings[e1],
                                secondary_embeddings[e2])
      if dist < best_dist:
        best_dist = dist
        best_e1 = e1
    result[e2] = best_e1
  return result


def find_closest_embeddings_multiple(path, thresh=0.01):
  primary_embeddings, secondary_embeddings = _load_embeddings(path)
  result = collections.defaultdict(list)
  for e2 in secondary_embeddings:
    best_dist = 1_000_000
    best_e1 = None
    for e1 in primary_embeddings:
      dist = embedding_distance(primary_embeddings[e1],
                                secondary_embeddings[e2])
      if dist < best_dist:
        best_dist = dist
        best_e1 = e1
    for e1 in primary_embeddings:
      dist = embedding_distance(primary_embeddings[e1],
                                secondary_embeddings[e2])
      if dist < best_dist + thresh:
        result[e2].append(e1)
  return result


def main(argv):
  result = find_closest_embeddings(FLAGS.embeddings)
  for e2 in result:
    print(f"{e2}\t{result[e2]}")


if __name__ == "__main__":
  app.run(main)
  

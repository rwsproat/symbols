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
"""Computes phonetic distances to create phonetic embeddings.
"""

import collections
import numpy as np
import random
import sys

import phon_dist

from absl import app
from absl import flags

flags.DEFINE_string("phoneme_classes", "Grm/phoneme_classes.tsv",
                    "File defining phoneme_classes")
flags.DEFINE_string("phone_list", "phon.txt",
                    "File with list of randomly generated "
                    "population of phonological forms of morphs")
flags.DEFINE_string("phone_embeddings", "phon_emb.tsv",
                    "Path to output with phone embeddings")
flags.DEFINE_string("phone_distances", "phon_dist.tsv",
                    "Path to output with phone distances")
flags.DEFINE_float("sample_proportion", 1.0,
                    "Proportion of cases to sample for distance")

FLAGS = flags.FLAGS


def read_phon_txt():
  hist = collections.defaultdict(int)
  morphs = set()
  max_nsyls_per_morph = 0
  with open(FLAGS.phone_list) as s:
    for line in s:
      morph = line.strip()
      morphs.add(morph)
      syls = morph.split(phon_dist.SYL_DIVIDER)
      if len(syls) > 2:  # Can only handle maximally disyllables for now.
        raise Exception(f"{morph} has {len(syls)}>2 syllables")
      if len(syls) > max_nsyls_per_morph:
        max_nsyls_per_morph = len(syls)
      for syl in syls:
        hist[syl] += 1
  syllables = [s[0] for s in sorted(hist.items(),
                                    key=lambda i: i[1], reverse=True)]
  first_k = syllables[:300]
  return syllables, first_k, morphs, max_nsyls_per_morph


def main(unused_argv):
  pd = phon_dist.PhoneticDistance(FLAGS.phoneme_classes)
  syllables, first_k, morphs, max_nsyls_per_morph = read_phon_txt()
  values = {}
  # Make a 300-long vector with each entry being the distance of the
  # syllable from the ith of the first 300 most common syllables
  if max_nsyls_per_morph == 2:
    first_k = first_k[:150]
  for m in morphs:
    val = []
    if max_nsyls_per_morph == 1:
      s = m
      for f in first_k:
        dist = pd.phonetic_distance(s, f)
        val.append(dist)
    else:
      syls = m.split(phon_dist.SYL_DIVIDER)
      try:
        s0 = syls[0]
        s1 = syls[1]
      # If this is a monosyllable, copy to both positions in order to allow it
      # to match on the rhyme and/or onset.
      except IndexError:
        s0 = syls[0] 
        s1 = syls[0]
      for f in first_k:
        dist = pd.phonetic_distance(s0, f) if s0 else 0
        val.append(dist)
      for f in first_k:
        dist = pd.phonetic_distance(s1, f) if s1 else 0
        val.append(dist)
    val = np.array(val)
    # Scale to max value = 1
    val /= max(val)
    # Shift to be between -1 and 1
    val = val * 2 - 1
    values[m] = val
  sys.stderr.write(f"Writing phone embeddings to {FLAGS.phone_embeddings}...")
  with open(FLAGS.phone_embeddings, "w") as stream:
    for s in values:
      stream.write(f"{s}\t[")
      stream.write(", ".join([f"{x:04f}" for x in list(values[s])]))
      stream.write("]\n")
  sys.stderr.write("done\n")
  distances = []
  vkeys = list(values.keys())

  sys.stderr.write(f"Computing distances...")
  if FLAGS.sample_proportion < 1.0:
    nkeys = len(vkeys)
    outer_range = list(range(nkeys))
    random.shuffle(outer_range)
    outer_range = outer_range[:int(nkeys * FLAGS.sample_proportion)]
    inner_range = list(range(nkeys))
    random.shuffle(inner_range)
    inner_range = inner_range[:int(nkeys * FLAGS.sample_proportion)]
    for i in outer_range:
      s1 = vkeys[i]
      for j in inner_range:
        if i == j:
          continue
        s2 = vkeys[j]
        distances.append((np.linalg.norm(values[s1] - values[s2]),
                         s1, s2))
  else:
    for i in range(len(vkeys)):
      s1 = vkeys[i]
      for j in range(i + 1, len(vkeys)):
        s2 = vkeys[j]
        distances.append((np.linalg.norm(values[s1] - values[s2]),
                          s1, s2))
  distances.sort()
  sys.stderr.write(f"done\n")
  sys.stderr.write(f"Writing phone distances to {FLAGS.phone_distances}...")
  with open(FLAGS.phone_distances, "w") as stream:
    for (d, s1, s2) in distances:
      stream.write(f"{d}\t{s1}\t{s2}\n")
  sys.stderr.write(f"done\n")


if __name__ == "__main__":
  app.run(main)

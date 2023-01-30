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
"""Analyzes written forms created in each epoch.

Example output columns:

Pron
Sem
Output glyph/output glyph semantics
Semantic distance
Closest pron
Best phon distance
Closest semantics (tagged with "**" if this is the best glyph for that meaning)
Confidence (diff between 1st and 2nd best prediction)

E.g.:

xaw  @DUCT   🦀/(=@THYME)    1.20    fa      4.42    @MOUTH  1.8913
"""

import collections
import csv

from absl import app
from absl import flags

import corpus as c
import embedding_distance as e
import numpy as np

import kruskal

flags.DEFINE_string("test_output", None, "Test output")
flags.DEFINE_string("training_file", None, "Training file")
# Note that the following, if provided should be the same value as
# returned by phon_embeddings. We only need the latter if the former
# is too big to precompute.
flags.DEFINE_string("phon_dist", None,
                    "If provided use this to calculate "
                    "phonetic distances.")
flags.DEFINE_string("phon_embeddings", None,
                    "If provided use this to calculate "
                    "phonetic distances.")

FLAGS = flags.FLAGS


def load_training(stream):
  reader = csv.reader(stream,
                      delimiter="\t", quotechar='"',
                      quoting=csv.QUOTE_MINIMAL)
  training = list(reader)
  hdr = training[0]
  training = training[1:]
  offset = 0
  for offset in range(len(hdr)):
    if hdr[offset] == "commodity_glyph":
      glyph_offset = offset
    elif hdr[offset] == "commodity_words":
      words_offset = offset
    elif hdr[offset] == "commodity":
      semantics_offset = offset
  prons = collections.defaultdict(set)
  pglyphs = collections.defaultdict(set)
  semantics = collections.defaultdict(str)
  sglyphs = collections.defaultdict(str)
  for t in training:
    prons[t[glyph_offset]].add(t[words_offset])
    pglyphs[t[words_offset]].add(t[glyph_offset])
    semantics[t[glyph_offset]] = t[semantics_offset]
    sglyphs[t[semantics_offset]] = t[glyph_offset]
  return prons, pglyphs, semantics, sglyphs


def process_output(prons, pglyphs, semantics, sglyphs,
                   closest_embeddings, phonetic_distance,
                   phon_embeddings, which_file):

  global total_phon_dist
  global total_sem_dist
  global total_pred_glyphs
  total_phon_dist = 0
  total_sem_dist = 0
  total_pred_glyphs = 0

  def process_output_glyph(output_glyph, sem, pron,
                           exact_pmatch_glyph, unused_k, top_2_diff):
    global total_phon_dist
    global total_sem_dist
    global total_pred_glyphs
    training_prons = list(prons[output_glyph])
    output_glyph_semantics = semantics[output_glyph]
    if not output_glyph_semantics:
      return
    closest_semantics = closest_embeddings[sem]
    best_glyph_for_sem = sglyphs[closest_semantics]
    if not exact_pmatch_glyph:
      exact_pmatch_glyph = "NONE"
    tag = ""
    if output_glyph_semantics == closest_semantics:
      tag = " <- Chose closest"
    best_phon_dist = 100_000
    which_tpron = ""
    for tpron in training_prons:
      if phon_embeddings:
        dist = np.linalg.norm(phon_embeddings[pron] - phon_embeddings[tpron])
      elif phonetic_distance:
        dist = phonetic_distance[pron, tpron]
      else:
        dist, _ = kruskal.BestMatch(pron, tpron)
      if dist < best_phon_dist:
        best_phon_dist = dist
        which_tpron = tpron
    sem_dist = e.concept_distance(output_glyph_semantics,
                                  sem)
    if output_glyph:
      total_phon_dist += best_phon_dist
      total_pred_glyphs += 1
      total_sem_dist += sem_dist
    tag = "  "
    if output_glyph in best_glyph_for_sem:
      tag = "**"
    tag = ""
    if output_glyph_semantics == closest_semantics:
      tag = "**"
    print(f"{pron}\t{sem}\t{output_glyph}/(={output_glyph_semantics})\t"
          f"{sem_dist:0.2f}\t"
          f"{which_tpron}\t{best_phon_dist:0.2f}\t"
          f"{tag}{closest_semantics}\t{top_2_diff}")

  with open(which_file) as s:
    for line in s:
      # TODO(rws): Fix it so this doesn't go into the file.
      if line.startswith("Reading phone"):
        continue
      line = line.strip("\n")
      toks = line.split("\t")
      sem = toks[1]
      pron = toks[5]
      output = toks[8]
      top_2_diff = float(toks[9])
      exact_pmatch_glyph = " ".join(pglyphs[pron])
      try:
        _, output_glyphs = output.split("#")
      except ValueError:
        output_glyphs = ""
      #print("*" * 80)
      # TODO(rws): Need to rethink the stats here since we are now
      # counting for each glyph in output...
      k = 0
      for output_glyph in output_glyphs:
        process_output_glyph(output_glyph, sem, pron,
                             exact_pmatch_glyph, k, top_2_diff)
        k += 1
  print(f"Mean Phonetic Distance:\t{total_phon_dist / total_pred_glyphs}")
  print(f"Mean Semantic Distance:\t{total_sem_dist / total_pred_glyphs}")
  print(f"Total predictions:\t{total_pred_glyphs}")


def main(unused_argv):
  closest_embeddings = e.find_closest_embeddings(FLAGS.embeddings)
  phonetic_distance = collections.defaultdict(float)
  phon_embeddings = {}
  if FLAGS.phon_dist:
    with open(FLAGS.phon_dist) as stream:
      for line in stream:
        dist, s1, s2 = line.strip("\n").split("\t")
        dist = float(dist)
        phonetic_distance[s1, s2] = dist
        phonetic_distance[s2, s1] = dist
  elif FLAGS.phon_embeddings:
    with open(FLAGS.phon_embeddings) as s:
      for line in s:
        phon, emb = line.strip("\n").split("\t")
        phon_embeddings[phon] = np.array(eval(emb))
  with open(FLAGS.training_file) as stream:
    prons, pglyphs, semantics, sglyphs = load_training(stream)
  process_output(prons, pglyphs, semantics, sglyphs,
                 closest_embeddings, phonetic_distance,
                 phon_embeddings,
                 FLAGS.test_output)



if __name__ == "__main__":
  app.run(main)

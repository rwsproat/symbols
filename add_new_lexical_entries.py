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
"""Creates lexicon with new symbols for new concepts.

Assumes the output of analyze_output.py run on the decoding of the novel_test
data for both semantic-only and semantic+phonetic conditions.
"""

import collections
import csv

from absl import app
from absl import flags

flags.DEFINE_string("semantic", None, "Semantic file")
flags.DEFINE_string("phonetic", None, "Test file")
flags.DEFINE_float("confidence", 4.0, "Minimum confidence of previous model")
flags.DEFINE_float("phon_closeness", 1, "Maximum phonetic closeness to allow")
flags.DEFINE_float("phon_closeness_for_sp", 2,
                   "Maximum phonetic closeness to allow for SP compounds")
flags.DEFINE_float("sem_closeness", 1, "Maximum semantic closeness to allow")

FLAGS = flags.FLAGS


def load_analysis(f, semantic=False):
  glyphs = {}
  if not f:
    return glyphs
  with open(f) as s:
    for line in s:
      if "@" not in line:
        continue
      line = line.strip("\n").split("\t")
      confidence = float(line[-1])
      if semantic:
        scloseness = float(line[3])
        pcloseness = 0
      else:
        scloseness = 0
        pcloseness = float(line[5])
      if (confidence > FLAGS.confidence and
          pcloseness < FLAGS.phon_closeness_for_sp and
          scloseness < FLAGS.sem_closeness):
        phon = line[0]
        sem = line[1]
        graph = line[2].split("/")[0]
        glyphs[sem, phon] = graph, pcloseness
  return glyphs


def main(unused_argv):
  semantic_glyphs = load_analysis(FLAGS.semantic, semantic=True)
  phonetic_glyphs = load_analysis(FLAGS.phonetic)
  seen = set()
  entries = []
  for (sem, phon) in phonetic_glyphs:
    pglyph, pcloseness = phonetic_glyphs[sem, phon]
    tag = ""
    # Only keep this as "P" if it is close enough
    if pcloseness < FLAGS.phon_closeness:
      tag = "P"
    sglyph = ""
    # Only keep this as "SP" if pcloseness < phon_closeness_for_sp,
    # which means it would show up in phonetic_glyphs.
    if (sem, phon) in semantic_glyphs:
      sglyph, _ = semantic_glyphs[sem, phon]
      tag = "SP"
    if tag:
      entries.append(f"{sem}\t{sglyph}{pglyph}\t{phon}\t#{tag}")
      seen.add((sem, phon))
  for (sem, phon) in semantic_glyphs:
    if (sem, phon) in seen:
      continue
    sglyph, _ = semantic_glyphs[sem, phon]
    entries.append(f"{sem}\t{sglyph}\t{phon}\t#S")
  for e in sorted(entries):
    print(e)


if __name__ == "__main__":
  app.run(main)

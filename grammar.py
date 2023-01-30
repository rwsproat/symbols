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
"""Generates sequence of phrases of form NUM NP.
"""

import concepts
import random
import sys

from absl import app
from absl import flags


flags.DEFINE_string("concept_embeddings", "emb.tsv",
                    "concept embeddings file")
flags.DEFINE_integer("max_phonological_forms", 2,
                     "Maximum number of phonological forms for a concept.")
flags.DEFINE_string("phon", "phon.txt",
                    "Path to randomly generated phonetic forms.")
flags.DEFINE_string("lexicon_path", None, "Path to the saved lexicon.")
flags.DEFINE_integer("max_quantity", 999,
                     "Maximum value for a ledger entry, but not > 999.")


FLAGS = flags.FLAGS


WORD_SPACE = "#"


class Number:
  """Simple number generator."""

  REQUIRED = list(range(1, 10)) + [10, 100]

  def __init__(self, number_lexicon):
    assert FLAGS.max_quantity < 1000
    self._number_lexicon = {}
    for (factor, symbol, word) in number_lexicon:
      self._number_lexicon[factor] = symbol, word
    for factor in self.REQUIRED:
      if factor < FLAGS.max_quantity:
        if factor not in self._number_lexicon:
          print(f"missing factor {factor}")
          assert factor in self._number_lexicon

  def generate(self):
    # Don't overgenerate in the hundreds range
    ranges = [random.randint(1, 10),
             random.randint(11, 100),
             random.randint(100, 999)]
    value = random.choice(ranges)
    # Keep trying...
    while value > FLAGS.max_quantity:
      value = random.choice(ranges)
    hundreds = value // 100
    tens = value % 100 // 10
    units = value % 100 % 10
    words = []
    glyphs = []
    if hundreds:
      glyph, word = self._number_lexicon[100]
      glyphs.append(glyph * hundreds)
      if hundreds == 1:
        words.append(word)
      else:
        _, unit_word = self._number_lexicon[hundreds]
        words.append(unit_word)
        words.append(word)
    if tens:
      glyph, word = self._number_lexicon[10]
      glyphs.append(glyph * tens)
      if tens == 1:
        words.append(word)
      else:
        _, unit_word = self._number_lexicon[tens]
        words.append(unit_word)
        words.append(word)
    if units:
      glyph, word = self._number_lexicon[units]
      glyphs.append(glyph)
      words.append(word)
    # TODO(rws): Revisit output form here
    return value, "".join(glyphs), WORD_SPACE.join(words)


class Commodity:
  """Simple commodity generator."""

  def __init__(self, lexicon):
    self._lexicon = []
    self._extra_lexicon = []
    self._lexicon_table = {}
    for (sem, glyph, word) in lexicon:
      self._lexicon_table[sem] = glyph, word
      if glyph == concepts.DUMMY_SYMBOL:
        self._extra_lexicon.append((sem, glyph, word))
      else:
        self._lexicon.append((sem, glyph, word))

  def generate(self, use_extra=False):
    if use_extra:
      return random.choice(self._extra_lexicon)
    else:
      return random.choice(self._lexicon)


class Ledger:
  """A ledger of one or more commodity listings."""

  def __init__(self, numbers, commodities):
    self._number = numbers
    self._commodities = commodities

  def generate(self, num_commodities, use_extra=False):

    def word_or_choice(word_or_list):
      if isinstance(word_or_list, list):
        return random.choice(word_or_list)
      return word_or_list

    rows = []
    hdr = [("count", "commodity",
            "count_glyphs", "commodity_glyph",
            "count_words", "commodity_words")]
    for _ in range(num_commodities):
      number_value, number_glyph, number_word = self._number.generate()
      (commodity_value, commodity_glyph, commodity_word
         ) = self._commodities.generate(use_extra=use_extra)
      rows.append((str(number_value), commodity_value,
                   number_glyph, commodity_glyph,
                   word_or_choice(number_word),
                   word_or_choice(commodity_word)))
    return hdr + rows

  def get_sem(self, rows):
    return [f"{r[0]} {r[1]}" for r in rows]

  def get_glyphs(self, rows):
    return [f"{r[2]} {r[3]}" for r in rows]

  def get_words(self, rows):
    return [f"{r[4]} {r[5]}" for r in rows]


class Phon:
  """
  """
  def __init__(self):
    with open(FLAGS.phon) as s:
      self._phonological_shapes = [c.strip() for c in s.readlines()]

  def generate(self, max_phonological_forms, barred_forms=set()):
    result = set()
    n = random.randint(1, max_phonological_forms)
    i = 0
    while i < n:
      phon = random.choice(self._phonological_shapes)
      if phon in barred_forms:
        continue
      result.add(phon)
      i += 1
    return list(result)


class Lexicon:
  """
  """
  def __init__(self, phon, numbers_and_symbols,
               restore_from_file=False):
    self._phon = phon
    self._entries = []
    self._number_entries = []
    if restore_from_file:
      self.restore_lexicon()
      return
    all_concepts = []
    with open(FLAGS.concept_embeddings) as s:
      for line in s:
        all_concepts.append(line.split()[0])
    for concept in concepts.CONCEPTS:
      symbol = concepts.CONCEPTS[concept]
      phons = self._phon.generate(FLAGS.max_phonological_forms)
      self._entries.append((concept, symbol, phons))
    # barred_forms makes sure that numbers don't overlap with main
    # lexical forms or with each other
    barred_forms = set()
    for concept in all_concepts:
      if concept in concepts.CONCEPTS:
        continue
      phons = self._phon.generate(FLAGS.max_phonological_forms)
      self._entries.append((concept, concepts.DUMMY_SYMBOL, phons))
      for phon in phons:
        barred_forms.add(phon)
    for (number, symbol) in numbers_and_symbols:
      phon = self._phon.generate(1, barred_forms=barred_forms)[0]
      barred_forms.add(phon)
      self._number_entries.append((number, symbol, phon))
      self.save_lexicon()

  @property
  def entries(self): return self._entries

  @property
  def number_entries(self): return self._number_entries

  def save_lexicon(self):
    if FLAGS.lexicon_path is not None:
      with open(FLAGS.lexicon_path, "w") as s:
        for (concept, symbol, phons) in self._entries:
          s.write(f"{concept}\t{symbol}\t{' '.join(phons)}")
        s.write("# Numbers\n")
        for (concept, symbol, phon) in self._number_entries:
          s.write(f"{concept}\t{symbol}\t{phon}")

  def restore_lexicon(self):
    # TODO(rws): We need to rethink this at some point since there could be a
    # symbol for a concept with a particular pronunciation but not for the same
    # concept with a different pronunciation.
    #
    # But maybe this is OK as is, since we are already getting:
    # 4	@DAFFODIL	IV	🌷	mit	yow
    # 10	@DAFFODIL	X	🌷🌋	wem	bik
    if FLAGS.lexicon_path is not None:
      sys.stderr.write(f"Restoring lexicon from {FLAGS.lexicon_path}\n")
      has_symbol = set()
      tmp_entries = []
      with open(FLAGS.lexicon_path) as s:
        lines = s.readlines()
        i = 0
        nlines = len(lines)
        while i < nlines:
          line = lines[i]
          i += 1
          if line.startswith("# Numbers"):
            break
          toks = line.strip("\n").split("\t")
          concept, symbol, phons = toks[:3]
          tmp_entries.append((concept, symbol, phons.split()))
          if symbol != concepts.DUMMY_SYMBOL:
            has_symbol.add(concept)
        while i < nlines:
          line = lines[i]
          i += 1
          concept, symbol, phon = line.strip("\n").split("\t")
          concept = int(concept)
          self._number_entries.append((concept, symbol, phon))
      # Remove concepts with dummy symbol if those concepts also have a
      # non-dummy symbol.
      for (concept, symbol, phons) in tmp_entries:
        if symbol == concepts.DUMMY_SYMBOL and concept in has_symbol:
          continue
        self._entries.append((concept, symbol, phons))


class Language:
  """"""

  def __init__(self, lexicon):
    self.commodity_ = Commodity(lexicon.entries)
    self.number_ = Number(lexicon.number_entries)
    self.ledger_ = Ledger(self.number_, self.commodity_)

  @property
  def commodity(self): return self.commodity_

  @property
  def number(self): return self.number_

  @property
  def ledger(self): return self.ledger_


def generate_language(restore_from_file=False):
  if restore_from_file:
    lexicon = Lexicon(None, None, restore_from_file)
  else:
    phon = Phon()
    number_lexicon = [(1, "I"),
                      (2, "II"),
                      (3, "III"),
                      (4, "IV"),
                      (5, "V"),
                      (6, "VI"),
                      (7, "VII"),
                      (8, "VIII"),
                      (9, "IX"),
                      (10, "X"),
                      (100, "C")]
    lexicon = Lexicon(phon, number_lexicon)
  return Language(lexicon)


def main(unused_argv):
  language = generate_language()
  print(language.ledger.generate(4))


if __name__ == "__main__":
  app.run(main)

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
"""Corpus generation code
"""

import collections
import csv
import grammar
import math
import numpy as np
import os
import random


from absl import app
from absl import flags

flags.DEFINE_integer("num_train", 1000,
                     "Number of lines to generate in the training corpus")
flags.DEFINE_integer("num_test", 100,
                     "Number of lines to generate in the test corpus")
flags.DEFINE_integer("num_novel", 100,
                     "Number of lines to generate in the novel test corpus")
flags.DEFINE_integer("batch_size", 16, "batch size")
flags.DEFINE_string("data_dir", "data", "Data output directory")
flags.DEFINE_string("phone_embeddings", None,
                    "If given path to phone embeddings: in that case these "
                    "are used instead of training embeddings")

FLAGS = flags.FLAGS


WORD_SPACE = grammar.WORD_SPACE
PAD = "<pad>"
BOS = "<s>"
EOS = "</s>"
BNC_EMBEDDING_SIZE = 300


class SymbolTable:
  """
  """
  def __init__(self):
    self._symbol_to_label = {}
    self._label_to_symbol = {}
    self._max_label = 0
    self.add_symbol(PAD)
    self.add_symbol(BOS)
    self.add_symbol(EOS)

  def add_symbol(self, symbol):
    try:
      k = self.find(symbol)
      return k
    except KeyError:
      self._symbol_to_label[symbol] = self._max_label
      self._label_to_symbol[self._max_label] = symbol
      self._max_label += 1
      return self.find(symbol)

  def find(self, idx):
    try:
      return self._symbol_to_label[idx]
    except KeyError:
      return self._label_to_symbol[idx]

  def write(self, path):
    with open(path, "w") as s:
      for label in sorted(self._label_to_symbol.keys()):
        s.write(f"{self._label_to_symbol[label]}\t{label}\n")
      
  def read(self, path):
    with open(path) as s:
      for line in s:
        symbol, label = line.strip("\n").split("\t")
        label = int(label)
        self._label_to_symbol[label] = symbol
        self._symbol_to_label[symbol] = label
        if label > self._max_label:
          self._max_label = label
    self._max_label += 1

  @property
  def size(self): return self._max_label


DataBundle = collections.namedtuple("data_bundle",
                                    ["count_string",
                                     "count_tensor",
                                     "commodity_string",
                                     "commodity_tensor",
                                     "count_glyph_string",
                                     "count_glyph_tensor",
                                     "commodity_glyph_string",
                                     "commodity_glyph_tensor",
                                     "target_string",
                                     "target_tensor",
                                     "count_word_string",
                                     "count_word_tensor",
                                     "commodity_word_string",
                                     "commodity_word_tensor",
                                     "ndata",
                                     ])


class Corpus:
  """"""

  DELIMITER = "\t"
  QUOTECHAR = '"'

  def __init__(self):
    self._texts = []
    self._count_symbols = SymbolTable()
    self._glyph_symbols = SymbolTable()
    self._phonology_symbols = SymbolTable() 
    self._embeddings = {}
    with open(FLAGS.concept_embeddings) as s:
      for line in s:
        conc, emb = line.strip("\n").split("\t")
        self._embeddings[conc] = eval(emb)
    self._phone_embeddings = {}
    if FLAGS.phone_embeddings:
      print(f"Reading phone_embeddings from {FLAGS.phone_embeddings}")
      with open(FLAGS.phone_embeddings) as s:
        for line in s:
          phon, emb = line.strip("\n").split("\t")
          self._phone_embeddings[phon] = eval(emb)
      self._phone_embeddings["**PAD**"] = [0] * 300
    self._construct_new_symbols = True

  def generate_texts(self, ledger, ntexts, use_extra=False):
    self._texts = ledger.generate(ntexts, use_extra=use_extra)
    self._parse_texts()

  def write_texts_and_symbols(self, ofile):
    with open(ofile, "w") as csvfile:
      writer = csv.writer(csvfile, delimiter=self.DELIMITER,
                          quotechar=self.QUOTECHAR, quoting=csv.QUOTE_MINIMAL)
      for row in self._texts:
        writer.writerow(row)
    self._count_symbols.write(f"{FLAGS.data_dir}/count.sym")
    self._glyph_symbols.write(f"{FLAGS.data_dir}/glyph.sym")
    self._phonology_symbols.write(f"{FLAGS.data_dir}/phonology.sym")

  def read_texts_and_symbols(self, ifile):
    self._construct_new_symbols = False
    with open(ifile) as csvfile:
      reader = csv.reader(csvfile, delimiter=self.DELIMITER,
                          quotechar=self.QUOTECHAR, quoting=csv.QUOTE_MINIMAL)
      self._texts = list(reader)
    self._parse_texts()
    self._count_symbols.read(f"{FLAGS.data_dir}/count.sym")
    self._glyph_symbols.read(f"{FLAGS.data_dir}/glyph.sym")
    self._phonology_symbols.read(f"{FLAGS.data_dir}/phonology.sym")

  @property
  def glyph_symbols(self): return self._glyph_symbols

  @property
  def text_body(self): return self._texts[1:]

  @property
  def text_hdr(self): return self._texts[0]

  def _parse_texts(self):
    hdr = self._texts[0]
    body = self._texts[1:]
    self._offsets = {}  # Want this to fail if we get the wrong name.
    for offset in range(len(hdr)):
      self._offsets[hdr[offset]] = offset

  def get(self, name):
    offset = self._offsets[name]
    return [t[offset] for t in self._texts[1:]]

  def data_for_batching(self):

    ntexts = len(self._texts[1:])

    def find_max(lst):
      return max(len(c) for c in lst)

    def find_max_syls(lst):

      def flatten(p):
        val = []
        for w in p.split("#"):
          val.append(w)
        return val

      return max(len(flatten(w)) for w in lst)

    def make_padded_data(lst, length, symbols, postpad=True):
      pad = [symbols.find(PAD)] * length
      result = []
      for w in lst:
        if self._construct_new_symbols:
          labels = [symbols.add_symbol(c) for c in w]
        else:
          labels = [symbols.find(c) for c in w]
        labels.append(symbols.find(EOS))
        if postpad:
          labels = (labels + pad)[:length]
        else:
          labels = ((labels[-1::-1] + pad)[:length])[-1::-1]
        result.append(labels)
      return result

    count_pad_len = find_max(self.get("count")) + 1
    count_glyph_pad_len = find_max(self.get("count_glyphs")) + 1
    commodity_glyph_pad_len = find_max(self.get("commodity_glyph")) + 1
    count_words_pad_len = find_max(self.get("count_words")) + 1 
    commodity_words_pad_len = find_max(self.get("commodity_words")) + 1 
    count_syls_pad_len = find_max_syls(self.get("count_words"))
    commodity_syls_pad_len = find_max_syls(self.get("commodity_words"))
    ################
    count_glyph_string=self.get("count_glyphs")
    commodity_glyph_string=self.get("commodity_glyph")
    ndata = len(count_glyph_string)
    target_string = []
    for i in range(ndata):
      target_string.append(
        count_glyph_string[i] + "#" + commodity_glyph_string[i])

    def count_word_tensors():
      if self._phone_embeddings:
        result = []
        for word in self.get("count_words"):
          this_word = []
          for syls in word.split("#"):
            this_word.append(self._phone_embeddings[syls])
          while len(this_word) < count_syls_pad_len:
            this_word.append(self._phone_embeddings["**PAD**"])
          result.append(this_word)
      else:
        result = make_padded_data(
            self.get("count_words"),
            count_words_pad_len,
            self._phonology_symbols)
      return result

    def commodity_word_tensors():
      if self._phone_embeddings:
        result = []
        for word in self.get("commodity_words"):
          this_word = []
          for syls in word.split("#"):
            this_word.append(self._phone_embeddings[syls])
          while len(this_word) < commodity_syls_pad_len:
            this_word.append(self._phone_embeddings["**PAD**"])
          result.append(this_word)
      else:
        result = make_padded_data(
            self.get("commodity_words"),
            commodity_words_pad_len,
            self._phonology_symbols)
      return result

    bundle = DataBundle(count_string=self.get("count"),
                        count_tensor=make_padded_data(
                          self.get("count"), count_pad_len,
                          self._count_symbols, postpad=False),
                        commodity_string=self.get("commodity"),
                        commodity_tensor=[
                          [self._embeddings[c]] for c in self.get("commodity")],
                        count_glyph_string=count_glyph_string,
                        count_glyph_tensor=make_padded_data(
                          self.get("count_glyphs"),
                          count_glyph_pad_len,
                          self._glyph_symbols),
                        commodity_glyph_string=commodity_glyph_string,
                        commodity_glyph_tensor=make_padded_data(
                          self.get("commodity_glyph"),
                          commodity_glyph_pad_len,
                          self._glyph_symbols),
                        target_string=target_string,
                        target_tensor=make_padded_data(
                          target_string,
                          count_glyph_pad_len + commodity_glyph_pad_len,
                          self._glyph_symbols),
                        count_word_string=self.get("count_words"),
                        count_word_tensor=count_word_tensors(),
                        commodity_word_string=self.get("commodity_words"),
                        commodity_word_tensor=commodity_word_tensors(),
                        ndata=ndata)
    return bundle


def batch(data_bundle, shuffle=True, batch_size=-1):
  if batch_size == -1:
    batch_size = FLAGS.batch_size

  indices = list(range(data_bundle.ndata))
  extra_indices = []
  for i in range(batch_size - data_bundle.ndata % batch_size):
    extra_indices.append(random.choice(indices))
  indices += extra_indices
  if shuffle:
    random.shuffle(indices)
  nindices = len(indices)
  batch = 0
  i = 0

  def samp(lst, indices):
    return [lst[i] for i in indices]

  def asamp(lst, indices):
    return np.array(samp(lst, indices), dtype=float)

  while i < nindices:
    idx = indices[i:i + batch_size]
    new_bundle = DataBundle(
      count_string=samp(data_bundle.count_string, idx),
      count_tensor=asamp(data_bundle.count_tensor, idx),
      commodity_string=samp(data_bundle.commodity_string, idx),
      commodity_tensor=asamp(data_bundle.commodity_tensor, idx),
      count_glyph_string=samp(data_bundle.count_glyph_string, idx),
      count_glyph_tensor=asamp(data_bundle.count_glyph_tensor, idx),
      commodity_glyph_string=samp(data_bundle.commodity_glyph_string, idx),
      commodity_glyph_tensor=asamp(data_bundle.commodity_glyph_tensor, idx),
      target_string=samp(data_bundle.target_string, idx),
      target_tensor=asamp(data_bundle.target_tensor, idx),
      count_word_string=samp(data_bundle.count_word_string, idx),
      count_word_tensor=asamp(data_bundle.count_word_tensor, idx),
      commodity_word_string=samp(data_bundle.commodity_word_string, idx),
      commodity_word_tensor=asamp(data_bundle.commodity_word_tensor, idx),
      ndata=batch_size)  # Useful only for the strings
    yield batch, new_bundle
    i += batch_size
    batch += 1
  yield -1, None


def nbatches(data_bundle, batch_size=-1):
  if batch_size == -1:
    batch_size = FLAGS.batch_size
  return math.ceil(data_bundle.ndata / batch_size)


def create_data():
  language = grammar.generate_language(
    restore_from_file=FLAGS.lexicon_path)
  corpus = Corpus()
  corpus.generate_texts(language.ledger, FLAGS.num_train)
  corpus.data_for_batching()
  try:
    os.mkdir(FLAGS.data_dir)
  except:
    pass
  corpus.write_texts_and_symbols(f"{FLAGS.data_dir}/train.tsv")
  corpus.generate_texts(language.ledger, FLAGS.num_test)
  corpus.data_for_batching()
  corpus.write_texts_and_symbols(f"{FLAGS.data_dir}/test.tsv")
  corpus.generate_texts(language.ledger, FLAGS.num_test, use_extra=True)
  corpus.data_for_batching()
  corpus.write_texts_and_symbols(f"{FLAGS.data_dir}/novel_test.tsv")


def main(unused_argv):
  create_data()
  train_corpus = Corpus()
  train_corpus.read_texts_and_symbols(f"{FLAGS.data_dir}/train.tsv")
  batch_data = train_corpus.data_for_batching()
  test_corpus = Corpus()
  test_corpus.read_texts_and_symbols(f"{FLAGS.data_dir}/test.tsv")
  batch_data = test_corpus.data_for_batching()
  batcher = batch(batch_data, shuffle=False)
  batch_num, this_batch = next(batcher)
  while batch_num > -1:
    print(batch_num)
    print(this_batch.count_string)
    print(this_batch.count_tensor)
    print(this_batch.commodity_string)
    print(this_batch.commodity_tensor)
    print(this_batch.count_glyph_string)
    print(this_batch.count_glyph_tensor)
    print(this_batch.commodity_glyph_string)
    print(this_batch.commodity_glyph_tensor)
    print(this_batch.target_string)
    print(this_batch.target_tensor)
    print(this_batch.count_word_string)
    print(this_batch.count_word_tensor)
    print(this_batch.commodity_word_string)
    print(this_batch.commodity_word_tensor)
    batch_num, this_batch = next(batcher)
  print(nbatches(batch_data))


if __name__ == "__main__":
  app.run(main)
  

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
"""Define the RNN model.
"""
import os
import time

import corpus as c
import numpy as np
import sys
import tensorflow as tf

from absl import logging        
from absl import flags


flags.DEFINE_bool("use_phonology", False,
                  "Whether to use the input's phonology information")
flags.DEFINE_bool("use_graphs", True,
                  "Whether to use the input's graph information")
flags.DEFINE_bool("use_semantics", True,
                  "Whether to use the input's semantic embeddings")


FLAGS = flags.FLAGS


class Encoder(tf.keras.Model):
  def __init__(self, vocab_size,
               enc_units,
               batch_size=None,
               embedding_dim=c.BNC_EMBEDDING_SIZE):
    super(Encoder, self).__init__()
    self._batch_size = batch_size
    self._embedding_dim = embedding_dim
    self._enc_units = enc_units
    self._embedding = tf.keras.layers.Embedding(vocab_size,
                                                self._embedding_dim)
    self._gru = tf.keras.layers.GRU(self._enc_units,
                                    return_sequences=True,
                                    return_state=True,
                                    recurrent_initializer="glorot_uniform")
    # Setting these for the initial stages
    self._use_semantics = FLAGS.use_semantics
    self._use_graphs = FLAGS.use_graphs
    self._use_phonology = FLAGS.use_phonology
    initializer = tf.random_normal_initializer(mean=1., stddev=2.)

  def call(self, inp, hidden):
    cntw = inp.count_word_tensor
    cmdw = inp.commodity_word_tensor
    cnt = inp.count_tensor
    cmd = inp.commodity_tensor
    cntg = inp.count_glyph_tensor
    cmdg = inp.commodity_glyph_tensor
    if not self._use_phonology:
      cntw = tf.zeros_like(cntw)
      cmdw = tf.zeros_like(cmdw)
    if not self._use_semantics:
      cnt = tf.zeros_like(cnt)
      cmd = tf.zeros_like(cmd)
    if not self._use_graphs:
      cntg = tf.zeros_like(cntg)
      cmdg = tf.zeros_like(cmdg)
    # If phone embeddings are used, we do not pass these to the
    # embedding. They are not trainable.
    if not FLAGS.phone_embeddings:
      cntw = self._embedding(cntw)
      cmdw = self._embedding(cmdw)
    cnt = self._embedding(cnt)
    # cmd is not passed to the embedding and is not trainable since it
    # is already an embedding.
    cntg = self._embedding(cntg)
    cmdg = self._embedding(cmdg)
    x = tf.concat([cntw, cmdw, cnt, cmd, cntg, cmdg], 1)
    _, state = self._gru(x, initial_state = hidden)
    return x, state

  def initialize_hidden_state(self):
    return tf.zeros((self._batch_size, self._enc_units))

  def switch_semantics(self, value=True):
    self._use_semantics = value
    print(f"use_semantics is now {self._use_semantics}")

  def switch_graphs(self, value=True):
    self._use_graphs = value
    print(f"use_graphs is now {self._use_graphs}")

  def switch_phonology(self, value=True):
    self._use_phonology = value
    print(f"use_phonology is now {self._use_phonology}")


class BahdanauAttention(tf.keras.layers.Layer):
  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    self._W1 = tf.keras.layers.Dense(units)
    self._W2 = tf.keras.layers.Dense(units)
    self._V = tf.keras.layers.Dense(1)

  def call(self, query, values):
    # hidden shape == (batch_size, hidden size)
    # hidden_with_time_axis shape == (batch_size, 1, hidden size)
    # we are doing this to perform addition to calculate the score
    hidden_with_time_axis = tf.expand_dims(query, 1)

    # score shape == (batch_size, max_length, 1) we get 1 at the last axis
    # because we are applying score to self.V the shape of the tensor before
    # applying self.V is (batch_size, max_length, units)
    score = self._V(tf.nn.tanh(
        self._W1(values) + self._W2(hidden_with_time_axis)))

    # attention_weights shape == (batch_size, max_length, 1)
    attention_weights = tf.nn.softmax(score, axis=1)

    # context_vector shape after sum == (batch_size, hidden_size)
    context_vector = attention_weights * values
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights


class Decoder(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, dec_units):
    super(Decoder, self).__init__()
    self._dec_units = dec_units
    self._embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self._gru = tf.keras.layers.GRU(self._dec_units,
                                    return_sequences=True,
                                    return_state=True,
                                    recurrent_initializer="glorot_uniform")
    self._fc = tf.keras.layers.Dense(vocab_size)

    # used for attention
    self._attention = BahdanauAttention(self._dec_units)

  def call(self, x, hidden, enc_output):
    # enc_output shape == (batch_size, max_length, hidden_size)
    context_vector, attention_weights = self._attention(hidden, enc_output)

    # x shape after passing through embedding == (batch_size, 1, embedding_dim)
    x = self._embedding(x)

    # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
    x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

    # passing the concatenated vector to the GRU
    output, state = self._gru(x)

    # output shape == (batch_size * 1, hidden_size)
    output = tf.reshape(output, (-1, output.shape[2]))

    # output shape == (batch_size, vocab)
    x = self._fc(output)

    return x, state, attention_weights


class Seq2SeqModel(object):

  def __init__(self,
               output_symbols=None,
               largest_input_vocab_size=None,
               enc_units=c.BNC_EMBEDDING_SIZE,
               dec_units=c.BNC_EMBEDDING_SIZE,
               data_dir=".",
               name="model"):
    # The following two must be identical:
    self._embedding_dim = enc_units 
    self._enc_units = enc_units
    self._dec_units = dec_units
    self._optimizer = tf.keras.optimizers.Adam()
    self._loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
      from_logits=True, reduction="none")
    self._output_symbols = output_symbols
    self._output_vocab_size = self._output_symbols.size
    # Note that the actual output shape of the encoder call must be the
    # embedding dimension, so we just make the two equal for simplicity.
    self._encoder = Encoder(batch_size=FLAGS.batch_size,
                            vocab_size=largest_input_vocab_size,
                            enc_units=self._embedding_dim,
                            embedding_dim=self._embedding_dim)
    self._decoder = Decoder(self._output_vocab_size,
                            self._embedding_dim,
                            self._dec_units)
    self._name = name
    self._data_dir = data_dir
    self._checkpoint_dir = f"{self._data_dir}/training_checkpoints_{self._name}"
    self._checkpoint_prefix = os.path.join(self._checkpoint_dir, "ckpt")
    self._checkpoint = tf.train.Checkpoint(optimizer=self._optimizer,
                                           encoder=self._encoder,
                                           decoder=self._decoder)

  def _loss_function(self, real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss = self._loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss.dtype)
    loss *= mask
    return tf.reduce_mean(loss)

  # NB: This doesn't want to be a tf.function since that will slow
  # things down immensely.
  def _train_step(self, inp, enc_hidden):
    loss = 0
    with tf.GradientTape() as tape:
      targ = inp.target_tensor
      enc_output, enc_hidden = self._encoder(inp, enc_hidden)
      dec_hidden = enc_hidden
      dec_input = tf.expand_dims(
        [self._output_symbols.find("<s>")] * FLAGS.batch_size, 1)
      for t in range(targ.shape[1]):
        predictions, dec_hidden, _ = self._decoder(
          dec_input, dec_hidden, enc_output)
        loss += self._loss_function(targ[:, t], predictions)
        dec_input = tf.expand_dims(targ[:, t], 1)
    batch_loss = (loss / int(targ.shape[1]))
    variables = (self._encoder.trainable_variables +
                 self._decoder.trainable_variables)
    gradients = tape.gradient(loss, variables)
    self._optimizer.apply_gradients(zip(gradients, variables))
    return batch_loss

  def train(self, epochs=10, switch_phonology_at_epoch=-1):
    best_total_loss = 1000000
    corpus = c.Corpus()
    corpus.read_texts_and_symbols(f"{FLAGS.data_dir}/train.tsv")
    batch_data = corpus.data_for_batching()
    nbatches = c.nbatches(batch_data)
    for epoch in range(epochs):
      if switch_phonology_at_epoch == epoch:
        self._encoder.switch_phonology(True)
      start = time.time()
      total_loss = 0
      steps = 0
      batcher = c.batch(batch_data, shuffle=True)
      batch_num, inp = next(batcher)
      enc_hidden = self._encoder.initialize_hidden_state()
      while batch_num > -1:
        batch_loss = self._train_step(inp, enc_hidden)
        total_loss += batch_loss
        if batch_num % 10 == 0:
          print("Epoch {} Batch {} (/{}) Loss {:.4f}".format(
            epoch + 1,
            batch_num,
            nbatches,
            batch_loss.numpy()))
        steps += 1
        batch_num, inp = next(batcher)
      total_loss /= steps
      print("Epoch {} Loss {:.4f} (previous best total loss = {:.4f})".format(
          epoch + 1, total_loss, best_total_loss))
      if total_loss < best_total_loss:
        self._checkpoint.save(file_prefix = self._checkpoint_prefix)
        print("Saved checkpoint to {}".format(self._checkpoint_prefix))
        best_total_loss = total_loss
      print("Time taken for 1 epoch {} sec\n".format(
        time.time() - start))
    print("Best total loss: {:.4f}".format(best_total_loss))

  def eval(self, which="test", restore_checkpoint=False):

    if restore_checkpoint:
      sys.stderr.write(f"Restoring checkpoint from {self._checkpoint_dir}\n")
      self._checkpoint.restore(
        tf.train.latest_checkpoint(self._checkpoint_dir))

    corpus = c.Corpus()
    if which == "test":
      corpus.read_texts_and_symbols(f"{FLAGS.data_dir}/test.tsv")
    elif which == "novel_test":
      corpus.read_texts_and_symbols(f"{FLAGS.data_dir}/novel_test.tsv")
    elif which == "train":
      corpus.read_texts_and_symbols(f"{FLAGS.data_dir}/train.tsv")
    else:
      raise Exception(f"No such condition: {which}")
    batch_data = corpus.data_for_batching()

    def decode_syms(x, syms, joiner=""):
      result = []
      i = 0
      length = len(x)
      in_commodity_glyph = False
      target_diff = 0.0
      for (s0, _, diff) in x:
        s0 = syms.find(s0)
        if s0 in ["</s>", "<pad>"]:
          continue
        result.append(s0)
        if in_commodity_glyph:
          target_diff = diff
        elif s0 == "#":
          in_commodity_glyph = True
      result = joiner.join(result)
      return result, f"{diff:0.4f}"

    tot = 0
    cor = 0
    cglyph_cor = 0
    enc_hidden = self._encoder.initialize_hidden_state()
    nbatches = c.nbatches(batch_data)
    batcher = c.batch(batch_data, shuffle=False)
    batch_num, inp = next(batcher)
    while batch_num > -1:
      enc_output, dec_hidden = self._encoder(inp, enc_hidden)
      targ = inp.target_tensor
      dec_input = tf.expand_dims([self._output_symbols.find("<s>")], 0)
      result = []
      for t in range(targ.shape[1]):
        predictions, dec_hidden, _ = self._decoder(
          dec_input, dec_hidden, enc_output)
        top2, idx2 = tf.math.top_k(predictions[0], 2)
        top2 = top2.numpy()
        idx2 = idx2.numpy()
        pred0 = int(idx2[0])
        if pred0 == self._output_symbols.find("</s>"):
          break
        else:
          pred1 = int(idx2[1])
          diff = top2[0] - top2[1]
          result.append((pred0, pred1, diff))
        dec_input = tf.expand_dims([pred0], 0)
      targ = inp.target_string[0]
      result, conf = decode_syms(result, self._output_symbols, joiner="")
      if which == "novel_test":
        tag = ""
      else:
        tag = "*" * 3
        if targ == result:
          tag = " " * 3
          cor += 1
          cglyph_cor += 1
        elif targ[-1] == result[-1]:  # Just commodity glyph right
          tag = ">" * 3
          cglyph_cor += 1
        tot += 1
      output_string = (f"{inp.count_string[0]}\t"
                       f"{inp.commodity_string[0]}\t"
                       f"{inp.count_glyph_string[0]}\t"
                       f"{inp.commodity_glyph_string[0]}\t"              
                       f"{inp.count_word_string[0]}\t"
                       f"{inp.commodity_word_string[0]}\t"
                       f"→\t"
                       f"{targ}\t"
                       f"{tag}{result}\t{conf}\t")
      print(output_string)
      batch_num, inp = next(batcher)
    if not which=="novel_test":
      print(f"Tot cor:\t{tot}\t{cor}\t{cor/tot}")
      print(f"Comm cor:\t{tot}\t{cglyph_cor}\t{cglyph_cor/tot}")
 
  @property
  def checkpoint(self): return self._checkpoint

  @property
  def checkpoint_dir(self): return self._checkpoint_dir

  @property
  def output_symbols(self): return self._output_symbols

  @property
  def encoder(self): return self._encoder
  

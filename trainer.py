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


from absl import app
from absl import flags

import corpus as c
import model as m
import sys

flags.DEFINE_bool("train", False, "Train the model")
flags.DEFINE_bool("test", False, "Test the model")
flags.DEFINE_bool("test_on_train", False,
                  "Test the model on the training data")
flags.DEFINE_bool("test_on_novel_test", False,
                  "Test the model on the novel test data")
flags.DEFINE_integer("epochs", 10, "Number of epochs for training")
flags.DEFINE_string("model_name", "model", "Name for model")

FLAGS = flags.FLAGS


def main(argv):
  glyph_symbols = c.SymbolTable()
  glyph_symbols.read(f"{FLAGS.data_dir}/glyph.sym")
  count_symbols = c.SymbolTable() 
  count_symbols.read(f"{FLAGS.data_dir}/count.sym")
  phonology_symbols = c.SymbolTable() 
  phonology_symbols.read(f"{FLAGS.data_dir}/phonology.sym")
  largest_input_vocab_size = glyph_symbols.size
  if count_symbols.size > largest_input_vocab_size:
    largest_input_vocab_size = count_symbols.size
  if phonology_symbols.size > largest_input_vocab_size:
    largest_input_vocab_size = phonology_symbols.size
  model = m.Seq2SeqModel(output_symbols=glyph_symbols,
                         largest_input_vocab_size=largest_input_vocab_size,
                         name=FLAGS.model_name)
  if FLAGS.test:
    assert FLAGS.batch_size == 1
    model.eval(which="test", restore_checkpoint=True)
  elif FLAGS.test_on_novel_test:
    assert FLAGS.batch_size == 1
    model.eval(which="novel_test", restore_checkpoint=True)
  elif FLAGS.test_on_train:
    assert FLAGS.batch_size == 1
    model.eval(which="train", restore_checkpoint=True)
  elif FLAGS.train:
    model.train(epochs=FLAGS.epochs)
  

if __name__ == "__main__":
  app.run(main)
  

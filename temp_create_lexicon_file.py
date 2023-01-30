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
"""Creates a lexicon from training and test files.

TODO(rws): [Work in progress]
This is an afterthought since we should have saved the lexicons as part of our
construction of the original data. This will only work because we have
monomorphemic number names. In future use the new --lexicon_path flag with
grammar.py.
"""

import collections
import csv

from absl import app
from absl import flags

flags.DEFINE_string("train", None, "Training file")
flags.DEFINE_string("test", None, "Test file")
flags.DEFINE_string("novel_test", None, "Novel test file")

FLAGS = flags.FLAGS


def main(unused_argv):
  DELIMITER = "\t"
  QUOTECHAR = '"'

  def process_file(f, concepts, numbers):
    if not f:
      return
    with open(f) as csvfile:
      reader = csv.reader(csvfile, delimiter=DELIMITER,
                          quotechar=QUOTECHAR, quoting=csv.QUOTE_MINIMAL)
      init = True
      for row in reader:
        if init:
          init = False
          continue
        concepts[row[1], row[3]].add(row[5])
        numbers[int(row[0]), row[2]] = row[4]
        
  concepts = collections.defaultdict(set)
  numbers = {}
  process_file(FLAGS.train, concepts, numbers)
  process_file(FLAGS.test, concepts, numbers)
  process_file(FLAGS.novel_test, concepts, numbers)
  for (concept, symbol) in sorted(concepts.keys()):
    phons = " ".join(concepts[concept, symbol])
    print(f"{concept}\t{symbol}\t{phons}")
  print("# Numbers")
  for (concept, symbol) in sorted(numbers.keys()):
    phon = numbers[concept, symbol]
    print(f"{concept}\t{symbol}\t{phon}")


if __name__ == "__main__":
  app.run(main)
  

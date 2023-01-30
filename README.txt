INTRO

This is the code used in the simulations reported in Chapter 7 of

Richard Sproat. Symbols: An Evolutionary History from the Stone Age to
the Future. Heidelberg, Springer, 2023.

DEPENDENCIES

. abseil
. numpy
. tensorflow (version 2)
. thrax (https://www.openfst.org/twiki/bin/view/GRM/Thrax) and dependencies

RUNNING THE CODE

To run a simulation through 5 epochs for the monosyllabic conditions:

./monosyllable.sh           # Epoch 0
./monosyllable_round01.sh   # Epoch 1
./monosyllable_round02.sh   # Epoch 2
./monosyllable_round03.sh   # Epoch 3
./monosyllable_round04.sh   # Epoch 4
./monosyllable_round05.sh   # Epoch 5

For the disyllable and sesquisyllable conditions, use the similarly
named disyllable.sh, disyllable_round0[1-5].sh, etc.

https://rws.xoba.com/symbols/rounds.tgz contains the dump of the main
data from the rounds reported in the book.

EMBEDDINGS

emb.tsv is derived from the British National Corpus embeddings. See:

Fares, Murhaf; Andrey Kutuzov; Stephan Oepen and Erik Vell- dal. 22-24
May 2017 2017. Word vectors, reuse, and replicability: Towards a
community repository of large-text resources. In Proceedings of the
21st Nordic Conference on Computational Linguistics, NoDaLiDa, pages
271– 276, Gothenburg.

LICENSE

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Author: Richard Sproat (rws@xoba.com)

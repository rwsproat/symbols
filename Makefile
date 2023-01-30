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
Grm/morphology.far: Grm/morphology.grm Grm/syllable.far Grm/byte.far
	thraxcompiler --input_grammar=$< --output_far=$@

Grm/syllable.far: Grm/syllable.grm 
	thraxcompiler --input_grammar=$< --output_far=$@

Grm/byte.far: Grm/byte.grm 
	thraxcompiler --input_grammar=$< --output_far=$@

clean:
	rm -f Grm/*.far


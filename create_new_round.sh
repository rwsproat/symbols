#!/bin/bash
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
source globalsettings.sh
COND=${COND:-monosyllable}
WHICH=${WHICH:-00}
ROUND=${ROUND:-01}
CONF=${CONF:-4}
PCLOSE=${PCLOSE:-1}
SPCLOSE=${SPCLOSE:-2}
SCLOSE=${SCLOSE:-1}
INDIR=${INDIR}
ROOTDIR="${DATADIR}/${COND}_data_${WHICH}"
OUTDIR="${DATADIR}/${COND}_data_${WHICH}_round${ROUND}"
# I.e. the original data on which this sequence of rounds is based
mkdir -p "${OUTDIR}"
for V in phonology nophonology
do
    python3 analyze_output.py \
	    --training_file="${INDIR}/train.tsv" \
	    --test_output="${INDIR}/results/novel_test_nograph_semantics_${V}.log" \
	    --phon_embeddings="${ROOTDIR}/phon_emb.tsv" >"${INDIR}/results/analysis_${V}.tsv"
done
python3 add_new_lexical_entries.py \
	--semantic="${INDIR}/results/analysis_nophonology.tsv" \
	--phonetic="${INDIR}/results/analysis_phonology.tsv" \
	--confidence="${CONF}" \
	--phon_closeness="${PCLOSE}" \
        --phon_closeness_for_sp="${SPCLOSE}" \
	--sem_closeness="${SCLOSE}" >"${OUTDIR}/lexicon.tsv"
python3 temp_create_lexicon_file.py \
	--train "${INDIR}/train.tsv" \
	--test "${INDIR}/test.tsv" \
	--novel_test="${INDIR}/novel_test.tsv" >>"${OUTDIR}/lexicon.tsv"
python3 corpus.py \
	--lexicon_path="${OUTDIR}/lexicon.tsv" \
	--num_train=5_000 \
	--num_test=500 \
	--num_novel=500 \
	--max_quantity=10 \
	--data_dir="${OUTDIR}" \
	--concept_embeddings="emb.tsv" \
	--phon="${ROOTDIR}/phon.txt" \
	--phone_embeddings="${ROOTDIR}/phon_emb.tsv"

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
ROUND="02"
NUM="${NUM:-00}"
RULE=CORE_DISYLLABLE
ODIR="${DATADIR}/disyllable_data_${NUM}"
DIR="${DATADIR}/disyllable_data_${NUM}_round${ROUND}"
#
COND=disyllable \
    WHICH="${NUM}" \
    ROUND="${ROUND}" \
    CONF=4 \
    PCLOSE=1 \
    SCLOSE=1 \
    INDIR="${DATADIR}/disyllable_data_${NUM}_round01" \
    ./create_new_round.sh
#
mkdir -p "${DIR}/results"
##
EMB_ARGS="--phone_embeddings=${ODIR}/phon_emb.tsv --data_dir=${DIR}"
COMMON_ARGS="--epochs=400 --alsologtostderr --batch_size=128 ${EMB_ARGS}"
###############################################################################
# Semantics only
echo COMMON_ARGS is "${COMMON_ARGS}"
TAG=nograph_semantics_nophonology
MODEL="model_${TAG}_${RULE}_${NUM}_round${ROUND}"
AUX_ARGS="--nouse_graphs --use_semantics --nouse_phonology"
python3 trainer.py --train ${COMMON_ARGS} ${AUX_ARGS} --model_name=${MODEL} |\
  stdbuf -o0 tee ${DIR}/results/train_${TAG}.log
python3 trainer.py --test --batch_size=1 ${AUX_ARGS} --model_name=${MODEL} ${EMB_ARGS} \
   >${DIR}/results/test_${TAG}.log
python3 trainer.py --test_on_novel_test --batch_size=1 ${AUX_ARGS} --model_name=${MODEL} \
   ${EMB_ARGS} > ${DIR}/results/novel_test_${TAG}.log
./clean_training_checkpoint.sh "${DIR}" "${MODEL}"
###############################################################################
# With phonology
echo COMMON_ARGS is "${COMMON_ARGS}"
TAG=nograph_semantics_phonology
MODEL="model_${TAG}_${RULE}_${NUM}_round${ROUND}"
AUX_ARGS="--nouse_graphs --use_semantics --use_phonology"
python3 trainer.py --train ${COMMON_ARGS} ${AUX_ARGS} --model_name=${MODEL} |\
  stdbuf -o0 tee ${DIR}/results/train_${TAG}.log
python3 trainer.py --test --batch_size=1 ${AUX_ARGS} --model_name=${MODEL} ${EMB_ARGS} \
   >${DIR}/results/test_${TAG}.log
python3 trainer.py --test_on_novel_test --batch_size=1 ${AUX_ARGS} --model_name=${MODEL} \
   ${EMB_ARGS} > ${DIR}/results/novel_test_${TAG}.log
./clean_training_checkpoint.sh "${DIR}" "${MODEL}"

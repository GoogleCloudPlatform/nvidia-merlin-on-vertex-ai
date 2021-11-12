#!/bin/bash
# Copyright 2021 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#            http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/bin/bash

# Set up a global error handler
err_handler() {
    echo "Error on line: $1"
    echo "Caused by: $2"
    echo "That returned exit status: $3"
    echo "Aborting..."
    exit $3
}

trap 'err_handler "$LINENO" "$BASH_COMMAND" "$?"' ERR


if [ -z "${AIP_STORAGE_URI}" ]
  then
    echo 'AIP_STORAGE_URI not set. Exiting ....'
    exit 1
fi

if [ -z "$1" ]
  then
    MODEL_REPOSITORY=/models
  else
    MODEL_REPOSITORY=$1
fi

    
echo "Copying model ensemble from ${AIP_STORAGE_URI} to ${MODEL_REPOSITORY}"
mkdir ${MODEL_REPOSITORY} 
gsutil -m cp -r ${AIP_STORAGE_URI}/* ${MODEL_REPOSITORY}

# gsutil does not copy empty dirs so create a version folder for the ensemble
ENSEMBLE_DIR=$(ls ${MODEL_REPOSITORY} | grep ens)
mkdir ${MODEL_REPOSITORY}/${ENSEMBLE_DIR}/1 

echo "Starting Triton Server"
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2 tritonserver --model-repository=$MODEL_REPOSITORY \
--backend-config=hugectr,ps=$MODEL_REPOSITORY/ps.json 
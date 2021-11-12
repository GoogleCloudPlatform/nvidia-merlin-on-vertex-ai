# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Exporting Triton ensemble model."""

import json
import os
import pathlib
from google.protobuf import text_format

import nvtabular as nvt
from nvtabular.inference.triton import export_hugectr_ensemble

# Temporary fix to https://github.com/NVIDIA-Merlin/NVTabular/issues/1221
import nvtabular.inference.triton.model_config_pb2 as model_config

HUGECTR_CONFIG_FILENAME = 'ps.json'


def create_hugectr_backend_config(
    model_path,
    model_repository_path='/models'):
  """Creates configurations definition for HugeCTR backend."""

  p = pathlib.Path(model_path)
  model_version = p.parts[-1]
  model_name = p.parts[-2]
  model_path_in_repository = os.path.join(model_repository_path, model_name,
                                          model_version)

  dense_pattern = f'{model_name}_dense_*.model'
  dense_path = [os.path.join(model_path_in_repository, path.name)
                for path in p.glob(dense_pattern)][0]
  sparse_pattern = f'{model_name}[0-9]_sparse_*.model'
  sparse_paths = [os.path.join(model_path_in_repository, path.name)
                  for path in p.glob(sparse_pattern)]
  network_file = os.path.join(model_path_in_repository, f'{model_name}.json')

  config_dict = dict()
  config_dict['supportlonglong'] = True
  model = dict()
  model['model'] = model_name
  model['sparse_files'] = sparse_paths
  model['dense_file'] = dense_path
  model['network_file'] = network_file
  config_dict['models'] = [model]

  return config_dict


def export_ensemble(
    model_name,
    workflow_path,
    saved_model_path,
    output_path,
    categorical_columns,
    continuous_columns,
    label_columns,
    num_slots,
    max_nnz,
    num_outputs,
    embedding_vector_size,
    max_batch_size,
    model_repository_path='/models'
):
  """Exports ensemble of models."""
  workflow = nvt.Workflow.load(workflow_path)

  hugectr_params = dict()
  graph_filename = f'{model_name}.json'
  hugectr_params['config'] = os.path.join(
      model_repository_path,
      model_name,
      '1',
      graph_filename)

  hugectr_params['slots'] = num_slots
  hugectr_params['max_nnz'] = max_nnz
  hugectr_params['embedding_vector_size'] = embedding_vector_size
  hugectr_params['n_outputs'] = num_outputs

  export_hugectr_ensemble(
      workflow=workflow,
      hugectr_model_path=saved_model_path,
      hugectr_params=hugectr_params,
      name=model_name,
      output_path=output_path,
      label_columns=label_columns,
      cats=categorical_columns,
      conts=continuous_columns,
      max_batch_size=max_batch_size,
  )

  hugectr_backend_config = create_hugectr_backend_config(
      model_path=os.path.join(output_path, model_name, '1'),
      model_repository_path=model_repository_path)

  with open(os.path.join(output_path, HUGECTR_CONFIG_FILENAME), 'w') as f:
    json.dump(hugectr_backend_config, f)

  # Temporary fix to https://github.com/NVIDIA-Merlin/NVTabular/issues/1221
  nvt_protobuf_path = os.path.join(output_path, f'{model_name}_nvt',
                                   'config.pbtxt')
  with open(nvt_protobuf_path, 'r') as f:
    nvt_config_pbtxt = f.read()
  nvt_config = model_config.ModelConfig()
  nvt_config = text_format.Parse(nvt_config_pbtxt, nvt_config)
  # nvt_config.instance_group[0].kind = 1
  nvt_config.ClearField('instance_group')
  with open(nvt_protobuf_path, 'w') as f:
    text_format.PrintMessage(nvt_config, f)

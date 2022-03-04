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
"""Training pipelines."""

import json
import time

from . import components
from . import config
from kfp.v2 import dsl

GKE_ACCELERATOR_KEY = 'cloud.google.com/gke-accelerator'


@dsl.pipeline(
    name=config.TRAINING_PIPELINE_NAME,
    pipeline_root=config.TRAINING_PIPELINE_ROOT
)
def training_pipeline(
    train_paths: list,
    valid_paths: list,
    num_output_files_train: int,
    num_output_files_valid: int,
    shuffle: str,
    per_gpu_batch_size: int,
    max_iter: int,
    max_eval_batches: int,
    eval_batches: int,
    dropout_rate: float,
    lr: float,
    num_epochs: int,
    eval_interval: int,
    snapshot: int,
    display_interval: int
):
  """Pipeline to train a HugeCTR model with data exported from GCS."""

  # ==================== Exporting tables as CSV ====================

  # === Convert train dataset from CSV to Parquet
  csv_to_parquet_train = components.convert_csv_to_parquet_op(
      data_paths=train_paths,
      split='train',
      num_output_files=num_output_files_train,
      n_workers=int(config.GPU_LIMIT),
      shuffle=shuffle
  )
  csv_to_parquet_train.set_display_name('Convert training split')
  csv_to_parquet_train.set_cpu_limit(config.CPU_LIMIT)
  csv_to_parquet_train.set_memory_limit(config.MEMORY_LIMIT)
  csv_to_parquet_train.set_gpu_limit(config.GPU_LIMIT)
  csv_to_parquet_train.add_node_selector_constraint(GKE_ACCELERATOR_KEY, config.GPU_TYPE)

  # === Convert eval dataset from CSV to Parquet
  csv_to_parquet_valid = components.convert_csv_to_parquet_op(
      data_paths=valid_paths,
      split='valid',
      num_output_files=num_output_files_valid,
      n_workers=int(config.GPU_LIMIT),
      shuffle=shuffle
  )
  csv_to_parquet_valid.set_display_name('Convert validation split')
  csv_to_parquet_valid.set_cpu_limit(config.CPU_LIMIT)
  csv_to_parquet_valid.set_memory_limit(config.MEMORY_LIMIT)
  csv_to_parquet_valid.set_gpu_limit(config.GPU_LIMIT)
  csv_to_parquet_valid.add_node_selector_constraint(GKE_ACCELERATOR_KEY, config.GPU_TYPE)

  # ==================== Analyse train dataset ==============================

  # === Analyze train data split
  analyze_dataset = components.analyze_dataset_op(
      parquet_dataset=csv_to_parquet_train.outputs['output_dataset'],
      n_workers=int(config.GPU_LIMIT)
  )
  analyze_dataset.set_display_name('Analyze')#.set_caching_options(enable_caching=True)
  analyze_dataset.set_cpu_limit(config.CPU_LIMIT)
  analyze_dataset.set_memory_limit(config.MEMORY_LIMIT)
  analyze_dataset.set_gpu_limit(config.GPU_LIMIT)
  analyze_dataset.add_node_selector_constraint(GKE_ACCELERATOR_KEY, config.GPU_TYPE)

  # ==================== Transform train and validation dataset =============

  # === Transform train data split
  transform_train = components.transform_dataset_op(
      workflow=analyze_dataset.outputs['workflow'],
      parquet_dataset=csv_to_parquet_train.outputs['output_dataset'],
      num_output_files=num_output_files_train,
      n_workers=int(config.GPU_LIMIT)
  )
  transform_train.set_display_name('Transform train split')
  transform_train.set_cpu_limit(config.CPU_LIMIT)
  transform_train.set_memory_limit(config.MEMORY_LIMIT)
  transform_train.set_gpu_limit(config.GPU_LIMIT)
  transform_train.add_node_selector_constraint(GKE_ACCELERATOR_KEY, config.GPU_TYPE)

  # === Transform eval data split
  transform_valid = components.transform_dataset_op(
      workflow=analyze_dataset.outputs['workflow'],
      parquet_dataset=csv_to_parquet_valid.outputs['output_dataset'],
      num_output_files=num_output_files_valid,
      n_workers=int(config.GPU_LIMIT)
  )
  transform_valid.set_display_name('Transform valid split')
  transform_valid.set_cpu_limit(config.CPU_LIMIT)
  transform_valid.set_memory_limit(config.MEMORY_LIMIT)
  transform_valid.set_gpu_limit(config.GPU_LIMIT)
  transform_valid.add_node_selector_constraint(GKE_ACCELERATOR_KEY, config.GPU_TYPE)

  # ==================== Train HugeCTR model ========================

  train_hugectr = components.train_hugectr_op(
      transformed_train_dataset=transform_train.outputs[
          'transformed_dataset'],
      transformed_valid_dataset=transform_valid.outputs[
          'transformed_dataset'],
      model_name=config.MODEL_NAME,
      project=config.PROJECT_ID,
      region=config.REGION,
      staging_location=config.STAGING_LOCATION,
      job_display_name=f'train-{config.MODEL_DISPLAY_NAME}-{time.strftime("%Y%m%d_%H%M%S")}',
      training_image_url=config.HUGECTR_IMAGE_URI,
      replica_count=int(config.REPLICA_COUNT),
      machine_type=config.MACHINE_TYPE,
      accelerator_type=config.ACCELERATOR_TYPE,
      accelerator_count=int(config.ACCELERATOR_NUM),
      num_workers=int(config.NUM_WORKERS),
      per_gpu_batch_size=per_gpu_batch_size,
      max_iter=max_iter,
      max_eval_batches=max_eval_batches,
      eval_batches=eval_batches,
      dropout_rate=dropout_rate,
      lr=lr,
      num_epochs=num_epochs,
      eval_interval=eval_interval,
      snapshot=snapshot,
      display_interval=display_interval
  )

  # ==================== Export Triton Model ==========================

  triton_ensemble = components.export_triton_ensemble(
      model=train_hugectr.outputs['model'],
      workflow=analyze_dataset.outputs['workflow'],
      model_name=config.MODEL_NAME,
      num_slots=int(config.NUM_SLOTS),
      max_nnz=int(config.MAX_NNZ),
      embedding_vector_size=int(config.EMBEDDING_VECTOR_SIZE),
      max_batch_size=int(config.MAX_BATCH_SIZE),
      model_repository_path=config.MODEL_REPOSITORY_PATH
  )

  # ==================== Upload to Vertex Models ======================

  components.upload_vertex_model(
      project=config.PROJECT_ID,
      region=config.REGION,
      display_name=config.MODEL_DISPLAY_NAME,
      exported_model=triton_ensemble.outputs['exported_model'],
      serving_container_image_uri=config.TRITON_IMAGE_URI
  )

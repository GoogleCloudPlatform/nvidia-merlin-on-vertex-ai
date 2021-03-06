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
"""Preprocessing pipelines."""

from . import components
from . import config
from kfp.v2 import dsl

GKE_ACCELERATOR_KEY = 'cloud.google.com/gke-accelerator'


@dsl.pipeline(
    name=config.PREPROCESS_CSV_PIPELINE_NAME,
    pipeline_root=config.PREPROCESS_CSV_PIPELINE_ROOT
)
def preprocessing_csv(
    train_paths: list,
    valid_paths: list,
    num_output_files_train: int,
    num_output_files_valid: int,
    shuffle: str
):
  """Pipeline to preprocess CSV files in GCS."""
  # ==================== Convert from CSV to Parquet ========================
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
  analyze_dataset.set_display_name('Analyze Dataset')#.set_caching_options(enable_caching=True)
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
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
from typing import Any
from . import components
from . import config
from kfp.v2 import dsl

GKE_ACCELERATOR_KEY = 'cloud.google.com/gke-accelerator'


@dsl.pipeline(
    name=config.PREPROCESS_CSV_PIPELINE_NAME,
    pipeline_root=config.PREPROCESS_CSV_PIPELINE_ROOT
)
def preprocessing_csv(
    train_paths: list[Any],
    valid_paths: list[Any],
    sep: str,
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
      sep=sep,
      shuffle=shuffle,
      n_workers=int(config.GPU_LIMIT),
      num_output_files=num_output_files_train
  )
  csv_to_parquet_train.set_cpu_limit(config.CPU_LIMIT)
  csv_to_parquet_train.set_memory_limit(config.MEMORY_LIMIT)
  csv_to_parquet_train.set_gpu_limit(config.GPU_LIMIT)
  csv_to_parquet_train.add_node_selector_constraint(GKE_ACCELERATOR_KEY,
                                                    config.GPU_TYPE)

  # === Convert eval dataset from CSV to Parquet
  csv_to_parquet_valid = components.convert_csv_to_parquet_op(
      data_paths=valid_paths,
      split='valid',
      sep=sep,
      n_workers=int(config.GPU_LIMIT),
      num_output_files=num_output_files_valid
  )
  csv_to_parquet_valid.set_cpu_limit(config.CPU_LIMIT)
  csv_to_parquet_valid.set_memory_limit(config.MEMORY_LIMIT)
  csv_to_parquet_valid.set_gpu_limit(config.GPU_LIMIT)
  csv_to_parquet_valid.add_node_selector_constraint(GKE_ACCELERATOR_KEY,
                                                    config.GPU_TYPE)

  # ==================== Analyse train dataset ==============================

  # === Analyze train data split
  analyze_dataset = components.analyze_dataset_op(
      parquet_dataset=csv_to_parquet_train.outputs['output_dataset'],
      n_workers=int(config.GPU_LIMIT)
  )
  analyze_dataset.set_cpu_limit(config.CPU_LIMIT)
  analyze_dataset.set_memory_limit(config.MEMORY_LIMIT)
  analyze_dataset.set_gpu_limit(config.GPU_LIMIT)
  analyze_dataset.add_node_selector_constraint(GKE_ACCELERATOR_KEY,
                                               config.GPU_TYPE)

  # ==================== Transform train and validation dataset =============

  # === Transform train data split
  transform_train_dataset = components.transform_dataset_op(
      workflow=analyze_dataset.outputs['workflow'],
      parquet_dataset=csv_to_parquet_train.outputs['output_dataset'],
      n_workers=int(config.GPU_LIMIT)
  )
  transform_train_dataset.set_cpu_limit(config.CPU_LIMIT)
  transform_train_dataset.set_memory_limit(config.MEMORY_LIMIT)
  transform_train_dataset.set_gpu_limit(config.GPU_LIMIT)
  transform_train_dataset.add_node_selector_constraint(GKE_ACCELERATOR_KEY,
                                                       config.GPU_TYPE)

  # === Transform eval data split
  transform_valid_dataset = components.transform_dataset_op(
      workflow=analyze_dataset.outputs['workflow'],
      parquet_dataset=csv_to_parquet_valid.outputs['output_dataset'],
      n_workers=int(config.GPU_LIMIT)
  )
  transform_valid_dataset.set_cpu_limit(config.CPU_LIMIT)
  transform_valid_dataset.set_memory_limit(config.MEMORY_LIMIT)
  transform_valid_dataset.set_gpu_limit(config.GPU_LIMIT)
  transform_valid_dataset.add_node_selector_constraint(GKE_ACCELERATOR_KEY,
                                                       config.GPU_TYPE)


@dsl.pipeline(
    name=config.PREPROCESS_BQ_PIPELINE_NAME,
    pipeline_root=config.PREPROCESS_BQ_PIPELINE_ROOT
)
def preprocessing_bq(
    shuffle: str
):
  """Pipeline to preprocess data from BigQuery."""
  # ==================== Exporting tables as Parquet ========================

  # === Export train table as parquet
  export_train_from_bq = components.export_parquet_from_bq_op(
      bq_project=config.PROJECT_ID,
      bq_dataset_name=config.BQ_DATASET_NAME,
      bq_location=config.BQ_LOCATION,
      bq_table_name=config.BQ_TRAIN_TABLE_NAME,
      split='train'
  )
  export_train_from_bq.set_cpu_limit(config.CPU_LIMIT)
  export_train_from_bq.set_memory_limit(config.MEMORY_LIMIT)

  # === Export valid table as parquet
  export_valid_from_bq = components.export_parquet_from_bq_op(
      bq_project=config.PROJECT_ID,
      bq_dataset_name=config.BQ_DATASET_NAME,
      bq_location=config.BQ_LOCATION,
      bq_table_name=config.BQ_VALID_TABLE_NAME,
      split='valid'
  )
  export_valid_from_bq.set_cpu_limit(config.CPU_LIMIT)
  export_valid_from_bq.set_memory_limit(config.MEMORY_LIMIT)

  # ==================== Analyse train dataset ==============================

  # === Analyze train data split
  analyze_dataset = components.analyze_dataset_op(
      parquet_dataset=export_train_from_bq.outputs['output_dataset'],
      n_workers=int(config.GPU_LIMIT)
  )
  analyze_dataset.set_cpu_limit(config.CPU_LIMIT)
  analyze_dataset.set_memory_limit(config.MEMORY_LIMIT)
  analyze_dataset.set_gpu_limit(config.GPU_LIMIT)
  analyze_dataset.add_node_selector_constraint(GKE_ACCELERATOR_KEY,
                                               config.GPU_TYPE)

  # ==================== Transform train and validation dataset =============

  # === Transform train data split
  transform_train_dataset = components.transform_dataset_op(
      workflow=analyze_dataset.outputs['workflow'],
      parquet_dataset=export_train_from_bq.outputs['output_dataset'],
      n_workers=int(config.GPU_LIMIT),
      shuffle=shuffle
  )
  transform_train_dataset.set_cpu_limit(config.CPU_LIMIT)
  transform_train_dataset.set_memory_limit(config.MEMORY_LIMIT)
  transform_train_dataset.set_gpu_limit(config.GPU_LIMIT)
  transform_train_dataset.add_node_selector_constraint(GKE_ACCELERATOR_KEY,
                                                       config.GPU_TYPE)

  # === Transform eval data split
  transform_valid_dataset = components.transform_dataset_op(
      workflow=analyze_dataset.outputs['workflow'],
      parquet_dataset=export_valid_from_bq.outputs['output_dataset'],
      n_workers=int(config.GPU_LIMIT),
      shuffle=shuffle
  )
  transform_valid_dataset.set_cpu_limit(config.CPU_LIMIT)
  transform_valid_dataset.set_memory_limit(config.MEMORY_LIMIT)
  transform_valid_dataset.set_gpu_limit(config.GPU_LIMIT)
  transform_valid_dataset.add_node_selector_constraint(GKE_ACCELERATOR_KEY,
                                                       config.GPU_TYPE)

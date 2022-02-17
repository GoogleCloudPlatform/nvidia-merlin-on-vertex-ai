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
def training_bq(
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
  """Pipeline to train a HugeCTR model with data exported from BQ."""
  # ==================== Exporting tables as Parquet ====================

  # === Export train table as parquet
  export_train_from_bq = components.export_parquet_from_bq_op(
      bq_project=config.PROJECT_ID,
      bq_dataset_name=config.BQ_DATASET_NAME,
      bq_location=config.BQ_LOCATION,
      bq_table_name=config.BQ_TRAIN_TABLE_NAME,
      split='train',
      instance_type=config.INSTANCE_TYPE,
      image_uri=config.NVT_IMAGE_URI,
      project_id=config.PROJECT_ID,
      region=config.REGION,
      workspace=config.WORKSPACE
  )

  # === Export valid table as parquet
  export_valid_from_bq = components.export_parquet_from_bq_op(
      bq_project=config.PROJECT_ID,
      bq_dataset_name=config.BQ_DATASET_NAME,
      bq_location=config.BQ_LOCATION,
      bq_table_name=config.BQ_TRAIN_TABLE_NAME,
      split='valid',
      instance_type=config.INSTANCE_TYPE,
      image_uri=config.NVT_IMAGE_URI,
      project_id=config.PROJECT_ID,
      region=config.REGION,
      workspace=config.WORKSPACE
  )

  # ==================== Analyse train dataset ==============================

  # === Analyze train data split
  analyze_dataset = components.analyze_dataset_op(
      parquet_dataset=export_train_from_bq.outputs['output_dataset'],
      n_workers=int(config.GPU_LIMIT),
      instance_type=config.INSTANCE_TYPE,
      gpu_type=config.GPU_TYPE,
      image_uri=config.NVT_IMAGE_URI,
      project_id=config.PROJECT_ID,
      region=config.REGION,
      workspace=config.WORKSPACE
  )

  # ==================== Transform train and validation dataset =============

  # === Transform train data split
  transform_train_dataset = components.transform_dataset_op(
      workflow=analyze_dataset.outputs['workflow'],
      parquet_dataset=export_train_from_bq.outputs['output_dataset'],
      n_workers=int(config.GPU_LIMIT),
      instance_type=config.INSTANCE_TYPE,
      gpu_type=config.GPU_TYPE,
      image_uri=config.NVT_IMAGE_URI,
      project_id=config.PROJECT_ID,
      region=config.REGION,
      workspace=config.WORKSPACE
  )

  # === Transform eval data split
  transform_valid_dataset = components.transform_dataset_op(
      workflow=analyze_dataset.outputs['workflow'],
      parquet_dataset=export_valid_from_bq.outputs['output_dataset'],
      n_workers=int(config.GPU_LIMIT),
      instance_type=config.INSTANCE_TYPE,
      gpu_type=config.GPU_TYPE,
      image_uri=config.NVT_IMAGE_URI,
      project_id=config.PROJECT_ID,
      region=config.REGION,
      workspace=config.WORKSPACE
  )

  # ==================== Train HugeCTR model ========================

  train_hugectr = components.train_hugectr_op(
      transformed_train_dataset=transform_train_dataset.outputs[
          'transformed_dataset'],
      transformed_valid_dataset=transform_valid_dataset.outputs[
          'transformed_dataset'],
      model_name=config.MODEL_NAME,
      project=config.PROJECT_ID,
      region=config.REGION,
      staging_location=config.STAGING_LOCATION,
      service_account=config.VERTEX_SA,
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

  labels = {
      "bq_dataset_name": config.BQ_DATASET_NAME,
      "bq_table_name": config.BQ_TRAIN_TABLE_NAME,
      "pipeline_name": config.TRAINING_PIPELINE_NAME,
      "pipeline_root": config.TRAINING_PIPELINE_ROOT
  }
  labels = json.dumps(labels)

  components.upload_vertex_model(
      project=config.PROJECT_ID,
      region=config.REGION,
      display_name=config.MODEL_DISPLAY_NAME,
      exported_model=triton_ensemble.outputs['exported_model'],
      serving_container_image_uri=config.TRITON_IMAGE_URI,
      labels=labels
  )

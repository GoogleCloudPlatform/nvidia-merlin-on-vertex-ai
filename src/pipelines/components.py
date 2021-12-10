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
"""KFP components."""

from typing import Optional
from . import config

from kfp.v2 import dsl
from kfp.v2.dsl import Artifact
from kfp.v2.dsl import Dataset
from kfp.v2.dsl import Input
from kfp.v2.dsl import Model
from kfp.v2.dsl import Output


@dsl.component(
    base_image=config.NVT_IMAGE_URI,
    install_kfp_package=False
)
def convert_csv_to_parquet_op(
    output_dataset: Output[Dataset],
    data_paths: list,
    split: str,
    sep: str,
    num_output_files: int,
    n_workers: int,
    shuffle: Optional[str] = None,
    recursive: Optional[bool] = False,
    device_limit_frac: Optional[float] = 0.8,
    device_pool_frac: Optional[float] = 0.9,
    part_mem_frac: Optional[float] = 0.125
):
  r"""Component to convert CSV file(s) to Parquet format using NVTabular.

  Args:
    output_dataset: Output[Dataset]
      Output metadata with references to the converted CSV files in GCS
      and the split name.The path to the files are in GCS fuse format:
      /gcs/<bucket name>/path/to/file
    data_paths: list
      List of paths to folders or files on GCS.
      For recursive folder search, set the recursive variable to True:
        'gs://<bucket_name>/<subfolder1>/<subfolder>/' or
        'gs://<bucket_name>/<subfolder1>/<subfolder>/flat_file.csv' or
        a combination of both.
    split: str
      Split name of the dataset. Example: train or valid
    sep: str
      CSV separator, example '\t'
    num_output_files: int
      Number of files to be exported during the conversion from CSV to Parquet
    n_workers: int
      Number of GPUs allocated to convert the CSV to Parquet
    shuffle: str
      How to shuffle the converted CSV, default to None. Options:
        PER_PARTITION
        PER_WORKER
        FULL
    recursive: bool
      Recursivelly search for files in path.
    device_limit_frac: Optional[float] = 0.8
    device_pool_frac: Optional[float] = 0.9
    part_mem_frac: Optional[float] = 0.125
  """
  from preprocessing.components import convert_csv_to_parquet_fn

  convert_csv_to_parquet_fn(
    output_dataset = output_dataset,
    data_paths = data_paths,
    split = split,
    sep = sep,
    num_output_files = num_output_files,
    n_workers = n_workers,
    shuffle = shuffle,
    recursive = recursive,
    device_limit_frac = device_limit_frac,
    device_pool_frac = device_pool_frac,
    part_mem_frac = part_mem_frac
  )


@dsl.component(
    base_image=config.NVT_IMAGE_URI,
    install_kfp_package=False
)
def analyze_dataset_op(
    parquet_dataset: Input[Dataset],
    workflow: Output[Artifact],
    n_workers: int,
    device_limit_frac: Optional[float] = 0.8,
    device_pool_frac: Optional[float] = 0.9,
    part_mem_frac: Optional[float] = 0.125
):
  """Component to generate statistics from the dataset.

  Args:
    parquet_dataset: Input[Dataset]
      Input metadata with references to the train and valid converted
      datasets in GCS and the split name.
    workflow: Output[Artifact]
      Output metadata with the path to the fitted workflow artifacts
      (statistics).
    n_workers: int
      Number of GPUs allocated to do the fitting process
    device_limit_frac: Optional[float] = 0.8
    device_pool_frac: Optional[float] = 0.9
    part_mem_frac: Optional[float] = 0.125
  """
  from preprocessing.components import analyze_dataset_fn

  analyze_dataset_fn(
    parquet_dataset = parquet_dataset,
    workflow = workflow,
    n_workers = n_workers,
    device_limit_frac = device_limit_frac,
    device_pool_frac = device_pool_frac,
    part_mem_frac = part_mem_frac
  )


@dsl.component(
    base_image=config.NVT_IMAGE_URI,
    install_kfp_package=False
)
def transform_dataset_op(
    workflow: Input[Artifact],
    parquet_dataset: Input[Dataset],
    transformed_dataset: Output[Dataset],
    n_workers: int,
    shuffle: str = None,
    device_limit_frac: float = 0.8,
    device_pool_frac: float = 0.9,
    part_mem_frac: float = 0.125
):
  """Component to transform a dataset according to the workflow definitions.

  Args:
    workflow: Input[Artifact]
      Input metadata with the path to the fitted_workflow
    parquet_dataset: Input[Dataset]
      Location of the converted dataset in GCS and split name
    transformed_dataset: Output[Dataset]
      Split name of the transformed dataset.
    n_workers: int
      Number of GPUs allocated to do the transformation
    shuffle: str
      How to shuffle the converted CSV, default to None. Options:
        PER_PARTITION
        PER_WORKER
        FULL
    device_limit_frac: float = 0.8
    device_pool_frac: float = 0.9
    part_mem_frac: float = 0.125
  """
  from preprocessing.components import transform_dataset_fn

  transform_dataset_fn(
    workflow = workflow,
    parquet_dataset = parquet_dataset,
    transformed_dataset = transformed_dataset,
    n_workers = n_workers,
    shuffle = shuffle,
    device_limit_frac = device_limit_frac,
    device_pool_frac = device_pool_frac,
    part_mem_frac = part_mem_frac
  )


@dsl.component(
    base_image=config.NVT_IMAGE_URI,
    install_kfp_package=False
)
def export_parquet_from_bq_op(
    output_dataset: Output[Dataset],
    bq_project: str,
    bq_location: str,
    bq_dataset_name: str,
    bq_table_name: str,
    split: str,
):
  """Component to export PARQUET files from a bigquery table.

  Args:
    output_dataset: dict
      Output metadata with the GCS path for the exported datasets.
    bq_project: str
      GCP project id: 'my_project'
    bq_location: str
      Location of the BQ dataset, example: 'US'
    bq_dataset_name: str
      Bigquery dataset name: 'my_dataset_name'
    bq_table_name: str
      Bigquery table name: 'my_train_table_id'
    split:
      Split name of the dataset. Example: train or valid
  """
  from preprocessing.components import export_parquet_from_bq_fn

  export_parquet_from_bq_fn(
    output_dataset = output_dataset,
    bq_project = bq_project,
    bq_location = bq_location,
    bq_dataset_name = bq_dataset_name,
    bq_table_name = bq_table_name,
    split = split,
  )


@dsl.component(
    base_image=config.NVT_IMAGE_URI,
    install_kfp_package=False
)
def train_hugectr_op(
    transformed_train_dataset: Input[Dataset],
    transformed_valid_dataset: Input[Dataset],
    model: Output[Model],
    model_name: str,
    project: str,
    region: str,
    staging_location: str,
    service_account: str,
    job_display_name: str,
    training_image_url: str,
    replica_count: int,
    machine_type: str,
    accelerator_type: str,
    accelerator_count: int,
    num_workers: int,
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
  """Trains a model with HugeCTR."""

  import logging
  import json
  import os
  from google.cloud import aiplatform as vertex_ai

  vertex_ai.init(
      project=project,
      location=region,
      staging_bucket=staging_location
  )

  train_data_fuse = os.path.join(transformed_train_dataset.path, 'train',
                                 '_file_list.txt').replace('gs://', '/gcs/')
  valid_data_fuse = os.path.join(transformed_valid_dataset.path, 'valid',
                                 '_file_list.txt').replace('gs://', '/gcs/')
  schema_path = os.path.join(transformed_train_dataset.path, 'train',
                             'schema.pbtxt').replace('gs://', '/gcs/')

  gpus = json.dumps([list(range(accelerator_count))]).replace(' ', '')

  worker_pool_specs = [
      {
          'machine_spec': {
              'machine_type': machine_type,
              'accelerator_type': accelerator_type,
              'accelerator_count': accelerator_count,
          },
          'replica_count': replica_count,
          'container_spec': {
              'image_uri': training_image_url,
              'command': ['python', '-m', 'task'],
              'args': [
                  f'--per_gpu_batch_size={per_gpu_batch_size}',
                  f'--model_name={model_name}',
                  f'--train_data={train_data_fuse}',
                  f'--valid_data={valid_data_fuse}',
                  f'--schema={schema_path}',
                  f'--max_iter={max_iter}',
                  f'--max_eval_batches={max_eval_batches}',
                  f'--eval_batches={eval_batches}',
                  f'--dropout_rate={dropout_rate}',
                  f'--lr={lr}',
                  f'--num_workers={num_workers}',
                  f'--num_epochs={num_epochs}',
                  f'--eval_interval={eval_interval}',
                  f'--snapshot={snapshot}',
                  f'--display_interval={display_interval}',
                  f'--gpus={gpus}',
              ],
          },
      }
  ]

  logging.info('worker_pool_specs:')
  logging.info(worker_pool_specs)

  logging.info('Submitting a custom job to Vertex AI...')
  job = vertex_ai.CustomJob(
      display_name=job_display_name,
      worker_pool_specs=worker_pool_specs,
      base_output_dir=model.uri
  )

  job.run(
      sync=True,
      service_account=service_account,
      restart_job_on_worker_restart=False
  )

  logging.info('Custom Vertex AI job completed.')


@dsl.component(
    base_image=config.NVT_IMAGE_URI,
    install_kfp_package=False
)
def export_triton_ensemble(
    model: Input[Model],
    workflow: Input[Artifact],
    exported_model: Output[Model],
    model_name: str,
    num_slots: int,
    max_nnz: int,
    embedding_vector_size: int,
    max_batch_size: int,
    model_repository_path: str
):
  """Exports model ensamble for prediction."""
  import logging
  from serving import export
  import feature_utils

  logging.info('Exporting Triton ensemble model...')
  export.export_ensemble(
      model_name=model_name,
      workflow_path=workflow.path,
      saved_model_path=model.path,
      output_path=exported_model.path,
      categorical_columns=feature_utils.categorical_columns(),
      continuous_columns=feature_utils.continuous_columns(),
      label_columns=feature_utils.label_columns(),
      num_slots=num_slots,
      max_nnz=max_nnz,
      num_outputs=len(feature_utils.label_columns()),
      embedding_vector_size=embedding_vector_size,
      max_batch_size=max_batch_size,
      model_repository_path=model_repository_path
  )
  logging.info('Triton model exported.')

@dsl.component(
    base_image=config.NVT_IMAGE_URI,
    install_kfp_package=False
)
def upload_vertex_model(
    exported_model: Input[Artifact],
    uploaded_model: Output[Artifact],
    project: str,
    region: str,
    display_name: str,
    serving_container_image_uri: str,
    serving_container_environment_variables: dict = dict(),
    labels: dict = dict()
):
  """Uploads model to vertex AI."""
  import logging
  from google.cloud import aiplatform as vertex_ai

  vertex_ai.init(project=project, location=region)

  exported_model_path = exported_model.path

  logging.info('Exported model location: %s', exported_model_path)

  vertex_model = vertex_ai.Model.upload(
      display_name=display_name,
      artifact_uri=exported_model_path,
      serving_container_image_uri=serving_container_image_uri,
      labels=labels
  )

  model_uri = vertex_model.gca_resource.name
  logging.info('Model uploaded to Vertex AI: %s', model_uri)
  uploaded_model.set_string_custom_property('model_uri', model_uri)

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
    num_output_files: int,
    n_workers: int,
    shuffle: Optional[str] = None,
    recursive: Optional[bool] = False,
    device_limit_frac: Optional[float] = 0.6,
    device_pool_frac: Optional[float] = 0.9,
    frac_size: Optional[float] = 0.10,
    memory_limit: Optional[int] = 100_000_000_000
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
    shuffle: str
      How to shuffle the converted CSV, default to None. Options:
        PER_PARTITION
        PER_WORKER
        FULL
    recursive: bool
      Recursivelly search for files in path.
    device_limit_frac: Optional[float] = 0.6
    device_pool_frac: Optional[float] = 0.9
    frac_size: Optional[float] = 0.10
    memory_limit: Optional[int] = 100_000_000_000
  """
  import os
  import logging

  from task import (
      create_cluster,
      create_csv_dataset,
      convert_csv_to_parquet,
      get_criteo_col_dtypes,
  )

  logging.info('Base path in %s', output_dataset.path)

  # Write metadata
  output_dataset.metadata['split'] = split

  logging.info('Creating cluster')
  create_cluster(
    n_workers=n_workers,
    device_limit_frac=device_limit_frac,
    device_pool_frac=device_pool_frac,
    memory_limit=memory_limit
  )
  
  logging.info(f'Creating CSV dataset from: {data_paths}')
  dataset = create_csv_dataset(
    data_paths=data_paths,
    recursive=recursive,
    col_dtypes=get_criteo_col_dtypes(),
    frac_size=frac_size
  )
  
  logging.info(f'Converting CSV to Parquet; {output_dataset.uri}')
  convert_csv_to_parquet(
    output_path=output_dataset.uri,
    dataset=dataset,
    output_files=num_output_files,
    shuffle=shuffle
  )


@dsl.component(
  base_image=config.NVT_IMAGE_URI,
  install_kfp_package=False
)
def analyze_dataset_op(
    parquet_dataset: Input[Dataset],
    workflow: Output[Artifact],
    n_workers: int,
    device_limit_frac: Optional[float] = 0.6,
    device_pool_frac: Optional[float] = 0.9,
    frac_size: Optional[float] = 0.10,
    memory_limit: Optional[int] = 100_000_000_000
):
  """Component to generate statistics from the dataset.

  Args:
    parquet_dataset: Input[Dataset]
      Input metadata with references to the train and valid converted
      datasets in GCS and the split name.
    workflow: Output[Artifact]
      Output metadata with the path to the fitted workflow artifacts
      (statistics).
    device_limit_frac: Optional[float] = 0.6
    device_pool_frac: Optional[float] = 0.9
    frac_size: Optional[float] = 0.10
  """
  import logging
  import nvtabular as nvt
  
  from task import (
      create_cluster,
      create_criteo_nvt_workflow,
  )

  logging.basicConfig(level=logging.INFO)

  create_cluster(
    n_workers=n_workers,
    device_limit_frac=device_limit_frac,
    device_pool_frac=device_pool_frac,
    memory_limit=memory_limit
  )

  logging.info('Creating Parquet dataset')
  dataset = nvt.Dataset(
      path_or_source=parquet_dataset.uri,
      engine='parquet',
      part_mem_fraction=frac_size
  )

  logging.info('Creating Workflow')
  # Create Workflow
  criteo_workflow = create_criteo_nvt_workflow()

  logging.info('Analyzing dataset')
  criteo_workflow = criteo_workflow.fit(dataset)

  logging.info('Saving Workflow')
  criteo_workflow.save(workflow.path)


@dsl.component(
  base_image=config.NVT_IMAGE_URI,
  install_kfp_package=False
)
def transform_dataset_op(
    workflow: Input[Artifact],
    parquet_dataset: Input[Dataset],
    transformed_dataset: Output[Dataset],
    num_output_files: int,
    n_workers: int,
    shuffle: str = None,
    device_limit_frac: float = 0.6,
    device_pool_frac: float = 0.9,
    frac_size: float = 0.10,
    memory_limit: int = 100_000_000_000
):
  """Component to transform a dataset according to the workflow definitions.

  Args:
    workflow: Input[Artifact]
      Input metadata with the path to the fitted_workflow
    parquet_dataset: Input[Dataset]
      Location of the converted dataset in GCS and split name
    transformed_dataset: Output[Dataset]
      Split name of the transformed dataset.
    shuffle: str
      How to shuffle the converted CSV, default to None. Options:
        PER_PARTITION
        PER_WORKER
        FULL
    device_limit_frac: float = 0.6
    device_pool_frac: float = 0.9
    frac_size: float = 0.10
  """
  import os
  import logging
  import nvtabular as nvt
  from merlin.schema import Tags
  
  from task import (
    create_cluster,
    save_dataset,
  )

  logging.basicConfig(level=logging.INFO)

  transformed_dataset.metadata['split'] = \
    parquet_dataset.metadata['split']
  
  logging.info('Creating cluster')
  create_cluster(
    n_workers=n_workers,
    device_limit_frac=device_limit_frac,
    device_pool_frac=device_pool_frac,
    memory_limit=memory_limit
  )

  logging.info(f'Creating Parquet dataset: {parquet_dataset.uri}')
  dataset = nvt.Dataset(
      path_or_source=parquet_dataset.uri,
      engine='parquet',
      part_mem_fraction=frac_size
  )

  logging.info('Loading Workflow')
  criteo_workflow = nvt.Workflow.load(workflow.path)

  logging.info('Transforming Dataset')
  trans_dataset = criteo_workflow.transform(dataset)

  logging.info(f'Saving transformed dataset: {transformed_dataset.uri}')
  save_dataset(
    dataset=trans_dataset,
    output_path=transformed_dataset.uri,
    output_files=num_output_files,
    shuffle=shuffle
  )

  logging.info('Generating file list for training.')
  file_list = os.path.join(transformed_dataset.path, '_file_list.txt')

  new_lines = []
  with open(file_list, 'r') as fp:
    lines = fp.readlines()
    new_lines.append(lines[0])
    for line in lines[1:]:
      new_lines.append(line.replace('gs://', '/gcs/'))

  gcs_file_list = os.path.join(transformed_dataset.path, '_gcs_file_list.txt')
  with open(gcs_file_list, 'w') as fp:
    fp.writelines(new_lines)

  logging.info('Saving cardinalities')
  cols_schemas = criteo_workflow.output_schema.select_by_tag(Tags.CATEGORICAL)
  cols_names = cols_schemas.column_names

  cards = []
  for c in cols_names:
    col = cols_schemas.get(c)
    cards.append(col.properties['embedding_sizes']['cardinality'])

  transformed_dataset.metadata['cardinalities'] = cards


@dsl.component(
  packages_to_install=[
    'google-cloud-aiplatform', 
    'google-cloud-pipeline-components',
    'kfp'
  ]
)
def train_hugectr_op(
    transformed_train_dataset: Input[Dataset],
    transformed_valid_dataset: Input[Dataset],
    model: Output[Model],
    model_name: str,
    project: str,
    region: str,
    staging_location: str,
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

  # cardinalities = ' '.join(
  #   [str(c) for c in transformed_train_dataset.metadata['cardinalities']]
  # )

  cardinalities = []
  for c in transformed_train_dataset.metadata['cardinalities']:
    if c > 9999999:
      cardinalities.append('9999999')
    else:
      cardinalities.append(str(c))

  cardinalities = ' '.join(cardinalities)

  train_data_fuse = os.path.join(
    transformed_train_dataset.path, '_gcs_file_list.txt'
  )
  valid_data_fuse = os.path.join(
    transformed_valid_dataset.path, '_gcs_file_list.txt'
  )
  schema_path = os.path.join(
    transformed_train_dataset.path, 'schema.pbtxt'
  )

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
                  f'--slot_size_array={cardinalities}',
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
  vertex_ai.init(
      project=project,
      location=region,
      staging_bucket=staging_location
  )

  job = vertex_ai.CustomJob(
      display_name=job_display_name,
      worker_pool_specs=worker_pool_specs,
      base_output_dir=model.uri
  )

  job.run(
      sync=True,
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
  import os
  from serving import export
  from serving import feature_utils

  logging.info('Exporting Triton ensemble model...')
  export.export_ensemble(
      model_name=model_name,
      workflow_path=workflow.path,
      saved_model_path=os.path.join(model.path, 'model'),
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
  packages_to_install=['google-cloud-aiplatform']
)
def upload_vertex_model(
    exported_model: Input[Artifact],
    uploaded_model: Output[Artifact],
    project: str,
    region: str,
    display_name: str,
    serving_container_image_uri: str
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
      serving_container_image_uri=serving_container_image_uri
  )

  model_uri = vertex_model.gca_resource.name
  logging.info('Model uploaded to Vertex AI: %s', model_uri)
  uploaded_model.metadata['model_uri'] = model_uri

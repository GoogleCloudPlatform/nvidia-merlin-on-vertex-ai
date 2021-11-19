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

from typing import Optional

from kfp.v2.dsl import Artifact
from kfp.v2.dsl import Dataset
from kfp.v2.dsl import Input
from kfp.v2.dsl import Model
from kfp.v2.dsl import Output


def convert_csv_to_parquet_fn(
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

  import logging
  import os
  from preprocessing import etl
  import feature_utils

  logging.basicConfig(level=logging.INFO)

  logging.info('Getting column names and dtypes')
  col_dtypes = feature_utils.get_criteo_col_dtypes()

  # Create Dask cluster
  logging.info('Creating Dask cluster.')
  client = etl.create_cluster(
      n_workers=n_workers,
      device_limit_frac=device_limit_frac,
      device_pool_frac=device_pool_frac
  )

  logging.info('Creating %s dataset.', split)
  dataset = etl.create_csv_dataset(
      data_paths=data_paths,
      sep=sep,
      recursive=recursive,
      col_dtypes=col_dtypes,
      part_mem_frac=part_mem_frac,
      client=client
  )

  logging.info('Base path in %s', output_dataset.path)
  fuse_output_dir = os.path.join(output_dataset.path, split)

  logging.info('Writing parquet file(s) to %s', fuse_output_dir)
  etl.convert_csv_to_parquet(
      output_path=fuse_output_dir,
      dataset=dataset,
      output_files=num_output_files,
      shuffle=shuffle
  )

  # Write metadata
  output_dataset.metadata['split'] = split


def analyze_dataset_fn(
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
  from preprocessing import etl
  import logging
  import os

  logging.basicConfig(level=logging.INFO)

  split = parquet_dataset.metadata['split']

  # Create Dask cluster
  logging.info('Creating Dask cluster.')
  client = etl.create_cluster(
      n_workers=n_workers,
      device_limit_frac=device_limit_frac,
      device_pool_frac=device_pool_frac
  )

  # Create data transformation workflow. This step will only
  # calculate statistics based on the transformations
  logging.info('Creating transformation workflow.')
  criteo_workflow = etl.create_criteo_nvt_workflow(client=client)

  # Create dataset to be fitted
  logging.info('Creating dataset to be analysed.')
  logging.info('Base path in %s', parquet_dataset.path)
  dataset = etl.create_parquet_dataset(
      client=client,
      data_path=os.path.join(
          parquet_dataset.path.replace('/gcs/', 'gs://'),
          split
      ),
      part_mem_frac=part_mem_frac
  )

  logging.info('Starting workflow fitting for %s split.', split)
  criteo_workflow = etl.analyze_dataset(criteo_workflow, dataset)
  logging.info('Finished generating statistics for dataset.')

  etl.save_workflow(criteo_workflow, workflow.path)
  logging.info('Workflow saved to GCS')


def transform_dataset_fn(
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
  from preprocessing import etl
  import logging
  import os

  logging.basicConfig(level=logging.INFO)

  # Create Dask cluster
  logging.info('Creating Dask cluster.')
  client = etl.create_cluster(
      n_workers=n_workers,
      device_limit_frac=device_limit_frac,
      device_pool_frac=device_pool_frac
  )

  logging.info('Loading workflow and statistics')
  criteo_workflow = etl.load_workflow(
      workflow_path=workflow.path,
      client=client
  )

  split = parquet_dataset.metadata['split']

  logging.info('Creating dataset definition for %s split', split)
  dataset = etl.create_parquet_dataset(
      client=client,
      data_path=os.path.join(
          parquet_dataset.path.replace('/gcs/', 'gs://'),
          split
      ),
      part_mem_frac=part_mem_frac
  )

  logging.info('Workflow is loaded')
  logging.info('Starting workflow transformation')
  dataset = etl.transform_dataset(
      dataset=dataset,
      workflow=criteo_workflow
  )

  logging.info('Applying transformation')
  etl.save_dataset(
      dataset, os.path.join(transformed_dataset.path, split), shuffle
  )

  transformed_dataset.metadata['split'] = split


def export_parquet_from_bq_fn(
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
  import logging
  import os
  from preprocessing import etl
  from google.cloud import bigquery

  logging.basicConfig(level=logging.INFO)

  client = bigquery.Client(project=bq_project)
  dataset_ref = bigquery.DatasetReference(bq_project, bq_dataset_name)

  full_output_path = os.path.join(
      output_dataset.path.replace('/gcs/', 'gs://'),
      split
  )

  logging.info(
      'Extracting %s table to %s path.', bq_table_name, full_output_path
  )
  etl.extract_table_from_bq(
      client=client,
      output_dir=full_output_path,
      dataset_ref=dataset_ref,
      table_id=bq_table_name,
      location=bq_location
  )

  # Write metadata
  output_dataset.metadata['split'] = split
  logging.info('Finished exporting to GCS.')

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
"""Data preprocessing."""

import os
from dask.distributed import Client
from dask_cuda import LocalCUDACluster
import fsspec
from google.cloud import bigquery
import nvtabular as nvt
from nvtabular.io.shuffle import Shuffle
from nvtabular.ops import Categorify
from nvtabular.ops import Clip
from nvtabular.ops import FillMissing
from nvtabular.ops import Normalize
from nvtabular.utils import device_mem_size


def create_csv_dataset(
    data_paths,
    sep,
    recursive,
    col_dtypes,
    part_mem_frac,
    client
):
  """Create nvt.Dataset definition for CSV files."""
  fs_spec = fsspec.filesystem('gs')
  rec_symbol = '**' if recursive else '*'

  valid_paths = []
  for path in data_paths:
    try:
      if fs_spec.isfile(path):
        valid_paths.append(path)
      else:
        path = os.path.join(path, rec_symbol)
        for i in fs_spec.glob(path):
          if fs_spec.isfile(i):
            valid_paths.append(f'gs://{i}')
    except FileNotFoundError as fnf_expt:
      print(fnf_expt)
      print('Incorrect path: {path}.')
    except OSError as os_err:
      print(os_err)
      print('Verify access to the bucket.')

    return nvt.Dataset(
        path_or_source=valid_paths,
        engine='csv',
        names=list(col_dtypes.keys()),
        sep=sep,
        dtypes=col_dtypes,
        part_size=int(part_mem_frac * device_mem_size()),
        client=client,
        assume_missing=True
    )


def convert_csv_to_parquet(
    output_path,
    dataset,
    output_files,
    shuffle=None
):
  """Convert CSV file to parquet and write to GCS."""
  if shuffle:
    shuffle = getattr(Shuffle, shuffle)

  dataset.to_parquet(
      output_path,
      # preserve_files=True,
      shuffle=shuffle,
      output_files=output_files
  )


def create_criteo_nvt_workflow(client):
  """Create a nvt.Workflow definition with transformation all the steps."""
  # Columns definition
  cont_names = ['I' + str(x) for x in range(1, 14)]
  cat_names = ['C' + str(x) for x in range(1, 27)]

  # Transformation pipeline
  num_buckets = 10000000
  categorify_op = Categorify(max_size=num_buckets)
  cat_features = cat_names >> categorify_op
  cont_features = cont_names >> FillMissing() >> Clip(
      min_value=0) >> Normalize()
  features = cat_features + cont_features + ['label']

  # Create and save workflow
  return nvt.Workflow(features, client)


def create_cluster(
    n_workers,
    device_limit_frac,
    device_pool_frac,
):
  """Create a Dask cluster to apply the transformations steps to the Dataset."""
  device_size = device_mem_size()
  device_limit = int(device_limit_frac * device_size)
  device_pool_size = int(device_pool_frac * device_size)
  rmm_pool_size = (device_pool_size // 256) * 256

  cluster = LocalCUDACluster(
      n_workers=n_workers,
      device_memory_limit=device_limit,
      rmm_pool_size=rmm_pool_size
  )

  return Client(cluster)


def create_parquet_dataset(
    client,
    data_path,
    part_mem_frac
):
  """Create a nvt.Dataset definition for the parquet files."""
  fs = fsspec.filesystem('gs')
  file_list = fs.glob(
      os.path.join(data_path, '*.parquet')
  )

  if not file_list:
    raise FileNotFoundError('Parquet file(s) not found')

  file_list = [os.path.join('gs://', i) for i in file_list]

  return nvt.Dataset(
      file_list,
      engine='parquet',
      part_size=int(part_mem_frac * device_mem_size()),
      client=client
  )


def analyze_dataset(
    workflow,
    dataset,
):
  """Calculate statistics for a given workflow."""
  workflow.fit(dataset)
  return workflow


def transform_dataset(
    dataset,
    workflow
):
  """Apply the transformations to the dataset."""
  workflow.transform(dataset)
  return dataset


def load_workflow(
    workflow_path,
    client,
):
  """Load a workflow definition from a path."""
  return nvt.Workflow.load(workflow_path, client)


def save_workflow(
    workflow,
    output_path
):
  """Save workflow to a path."""
  workflow.save(output_path)


def save_dataset(
    dataset,
    output_path,
    shuffle=None
):
  """Save dataset to parquet files to path."""
  if shuffle:
    shuffle = getattr(Shuffle, shuffle)

  dataset.to_parquet(
      output_path=output_path,
      shuffle=shuffle
  )


def extract_table_from_bq(
    client,
    output_dir,
    dataset_ref,
    table_id,
    location='us'
):
  """Create job to extract parquet files from BQ tables."""
  extract_job_config = bigquery.ExtractJobConfig()
  extract_job_config.destination_format = 'PARQUET'

  bq_glob_path = os.path.join(output_dir, 'criteo-*.parquet')
  table_ref = dataset_ref.table(table_id)

  extract_job = client.extract_table(
      table_ref,
      bq_glob_path,
      location=location,
      job_config=extract_job_config
  )
  extract_job.result()

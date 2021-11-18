def convert_csv_to_parquet_op(
    output_dataset,
    data_paths: list,
    split: str,
    sep: str,
    num_output_files: int,
    n_workers: int,
    shuffle = None,
    recursive = False,
    device_limit_frac = 0.8,
    device_pool_frac = 0.9,
    part_mem_frac = 0.125
):

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
  
  # Modified to return the output_dataset
  # This return is implicit on KFP Pipelines
  return output_dataset


def analyze_dataset_op(
    parquet_dataset, #: Input[Dataset]
    workflow, #: Output[Artifact]
    n_workers: int,
    device_limit_frac = 0.8,
    device_pool_frac = 0.9,
    part_mem_frac = 0.125
):
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

  return workflow


def transform_dataset_op(
    workflow, #: Input[Artifact]
    parquet_dataset, #: Input[Dataset]
    transformed_dataset, #: Output[Dataset]
    n_workers: int,
    shuffle: str = None,
    device_limit_frac: float = 0.8,
    device_pool_frac: float = 0.9,
    part_mem_frac: float = 0.125
):
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

  return transformed_dataset


def export_parquet_from_bq_op(
    output_dataset, #: Output[Dataset],
    bq_project: str,
    bq_location: str,
    bq_dataset_name: str,
    bq_table_name: str,
    split: str,
):
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

  return output_dataset

from preprocessing import components

# Standard Libraries
import argparse
import logging
from collections import namedtuple

# Container to store data passed to Pipeline steps
ComponentData = namedtuple(
    'ComponentData', ['metadata', 'path', 'uri']
)

def convert_csv_to_parquet_op(args):
    output_dataset = ComponentData(
        metadata = {},
        path = args.output_path,
        uri = ''
    )

    data_paths = [args.data_paths]

    logging.info(f'Starting job.')

    output_dataset = components.convert_csv_to_parquet_fn(
        output_dataset=output_dataset,
        data_paths=data_paths,
        split=args.split,
        sep=args.sep,
        num_output_files=args.num_output_files,
        n_workers=args.n_workers,
        shuffle=args.shuffle,
        recursive=args.recursive,
        device_limit_frac=args.device_limit_frac,
        device_pool_frac=args.device_pool_frac,
        part_mem_frac=args.part_mem_frac
    )

    return output_dataset


def analyze_dataset_op(args):
    parquet_dataset = ComponentData(
        metadata = {'split': args.split},
        path = args.output_path,
        uri = ''
    )

    workflow = ComponentData(
        metadata = {},
        path = args.workflow_path,
        uri = ''
    )

    logging.info(f'Starting job.')

    components.analyze_dataset_fn(
        parquet_dataset = parquet_dataset, #: Input[Dataset]
        workflow = workflow, #: Output[Artifact]
        n_workers = args.n_workers,
        device_limit_frac = args.device_limit_frac,
        device_pool_frac = args.device_pool_frac,
        part_mem_frac = args.part_mem_frac
    )

    return workflow


def transform_dataset_op(args):
    parquet_dataset = ComponentData(
        metadata = {'split': args.split},
        path = args.output_path,
        uri = ''
    )

    workflow = ComponentData(
        metadata = {},
        path = args.workflow_path,
        uri = ''
    )

    transformed_dataset = ComponentData(
        metadata = {},
        path = args.transformed_dataset,
        uri = ''
    )

    logging.info(f'Starting job.')
 
    return components.transform_dataset_fn(
        workflow = workflow,
        parquet_dataset = parquet_dataset,
        transformed_dataset = transformed_dataset,
        n_workers = args.n_workers,
        shuffle = args.shuffle,
        device_limit_frac = args.device_limit_frac,
        device_pool_frac = args.device_pool_frac,
        part_mem_frac = args.part_mem_frac
    )

def export_parquet_from_bq_op(args):
    output_dataset = ComponentData(
        metadata = {},
        path = args.output_path,
        uri = ''
    )

    logging.info(f'Starting job.')

    return components.export_parquet_from_bq_fn(
        output_dataset = output_dataset,
        bq_project = args.bq_project,
        bq_location = args.bq_location,
        bq_dataset_name = args.bq_dataset_name,
        bq_table_name = args.bq_table_name,
        split = args.split,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--method-to-call',
                        type=str,
                        default='')
    parser.add_argument('--output-path',
                        type=str,
                        default='')
    parser.add_argument('--data-paths',
                        type=str,
                        default='')
    parser.add_argument('--workflow-path',
                        type=str,
                        default='')
    parser.add_argument('--transformed-dataset',
                        type=str,
                        default='')
    parser.add_argument('--bq-project',
                        type=str,
                        default='')
    parser.add_argument('--bq-location',
                        type=str,
                        default='')
    parser.add_argument('--bq-dataset-name',
                        type=str,
                        default='')
    parser.add_argument('--bq-table-name',
                        type=str,
                        default='')
    parser.add_argument('--split',
                        type=str,
                        default='')
    parser.add_argument('--sep',
                        type=str,
                        default='\t')
    parser.add_argument('--num-output-files',
                        type=int,
                        default=1)
    parser.add_argument('--n-workers',
                        type=int,
                        default=1)
    parser.add_argument('--shuffle',
                        type=str,
                        default='None')
    parser.add_argument('--recursive',
                        type=bool,
                        default=False)
    parser.add_argument('--part-mem-frac',
                        type=float,
                        default=0.125)
    parser.add_argument('--device-limit-frac',
                        type=float,
                        default=0.8)
    parser.add_argument('--device-pool-frac',
                        type=float,
                        default=0.9)

    args = parser.parse_args()

    logging.basicConfig(
        format='%(asctime)s - %(message)s', 
        level=logging.INFO, 
        datefmt='%d-%m-%y %H:%M:%S'
    )

    logging.info(f"Args: {args}")

    method_to_call = globals()[args.method_to_call]
    method_to_call(args)

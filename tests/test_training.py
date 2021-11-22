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

from serving import export
import feature_utils

# Standard Libraries
import argparse
import logging


def export_ensemble_op(args):

    logging.info(f'Starting job.')

    output_dataset = export.export_ensemble(
      model_name=args.model_name,
      workflow_path=args.workflow_path,
      saved_model_path=args.saved_model_path,
      output_path=args.output_path,
      categorical_columns=feature_utils.categorical_columns(),
      continuous_columns=feature_utils.continuous_columns(),
      label_columns=feature_utils.label_columns(),
      num_slots=args.num_slots,
      max_nnz=args.max_nnz,
      num_outputs=args.num_outputs,
      embedding_vector_size=args.embedding_vector_size,
      max_batch_size=args.max_batch_size,
      model_repository_path=args.model_repository_path
    )

    return output_dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--method-to-call',
                        type=str,
                        default='')
    parser.add_argument('--model-name',
                        type=str,
                        default='')
    parser.add_argument('--workflow-path',
                        type=str,
                        default='')
    parser.add_argument('--saved-model-path',
                        type=str,
                        default='')
    parser.add_argument('--output-path',
                        type=str,
                        default='')
    parser.add_argument('--num-slots',
                        type=int,
                        default=1)
    parser.add_argument('--max-nnz',
                        type=int,
                        default=1)
    parser.add_argument('--num-outputs',
                        type=int,
                        default=1)
    parser.add_argument('--embedding-vector-size',
                        type=int,
                        default=1)
    parser.add_argument('--max-batch-size',
                        type=int,
                        default=1)
    parser.add_argument('--model-repository-path',
                        type=str,
                        default='')

    args = parser.parse_args()

    logging.basicConfig(
        format='%(asctime)s - %(message)s', 
        level=logging.INFO, 
        datefmt='%d-%m-%y %H:%M:%S'
    )

    logging.info(f"Args: {args}")

    method_to_call = globals()[args.method_to_call]
    method_to_call(args)

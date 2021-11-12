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
"""DeepFM Network in HugeCTR."""
from typing import List, Any
import hugectr
from mpi4py import MPI


def create_model(
    train_data: List[Any],
    valid_data: str,
    slot_size_array: List[Any],
    gpus: List[Any],
    max_eval_batches: int = 300,
    batchsize: int = 2048,
    lr: float = 0.001,
    dropout_rate: float = 0.5,
    workspace_size_per_gpu: float = 61,
    num_dense_features: int = 13,
    num_sparse_features: int = 26,
    nnz_per_slot: int = 2,
    num_workers: int = 12,
    repeat_dataset: bool = True,
):
  """DeepFM Network(https://www.ijcai.org/Proceedings/2017/0239.pdf)."""

  if not gpus:
    gpus = [[0]]

  solver = hugectr.CreateSolver(max_eval_batches=max_eval_batches,
                                batchsize_eval=batchsize,
                                batchsize=batchsize,
                                lr=lr,
                                vvgpu=gpus,
                                repeat_dataset=repeat_dataset,
                                i64_input_key=True)

  reader = hugectr.DataReaderParams(
      data_reader_type=hugectr.DataReaderType_t.Parquet,
      source=train_data,
      eval_source=valid_data,
      slot_size_array=slot_size_array,
      check_type=hugectr.Check_t.Non,
      num_workers=num_workers)

  optimizer = hugectr.CreateOptimizer(optimizer_type=hugectr.Optimizer_t.Adam,
                                     update_type=hugectr.Update_t.Global,
                                     beta1=0.9,
                                     beta2=0.999,
                                     epsilon=0.0000001)

  model = hugectr.Model(solver, reader, optimizer)

  model.add(hugectr.Input(
      label_dim=1, label_name="label",
      dense_dim=num_dense_features, dense_name="dense",
      data_reader_sparse_param_array=[hugectr.DataReaderSparseParam(
          "data1", nnz_per_slot, False, num_sparse_features)]))

  model.add(hugectr.SparseEmbedding(
      embedding_type=hugectr.Embedding_t.LocalizedSlotSparseEmbeddingHash,
      workspace_size_per_gpu_in_mb=workspace_size_per_gpu,
      embedding_vec_size=11,
      combiner="sum",
      sparse_embedding_name="sparse_embedding1",
      bottom_name="data1",
      slot_size_array=slot_size_array,
      optimizer=optimizer))

  model.add(hugectr.DenseLayer(layer_type=hugectr.Layer_t.Reshape,
                               bottom_names=["sparse_embedding1"],
                               top_names=["reshape1"],
                               leading_dim=11))

  model.add(hugectr.DenseLayer(layer_type=hugectr.Layer_t.Slice,
                               bottom_names=["reshape1"],
                               top_names=["slice11", "slice12"],
                               ranges=[(0, 10), (10, 11)]))

  model.add(hugectr.DenseLayer(layer_type=hugectr.Layer_t.Reshape,
                               bottom_names=["slice11"],
                               top_names=["reshape2"],
                               leading_dim=260))

  model.add(hugectr.DenseLayer(layer_type=hugectr.Layer_t.Reshape,
                               bottom_names=["slice12"],
                               top_names=["reshape3"],
                               leading_dim=26))

  model.add(hugectr.DenseLayer(layer_type=hugectr.Layer_t.Slice,
                               bottom_names=["dense"],
                               top_names=["slice21", "slice22"],
                               ranges=[(0, 13), (0, 13)]))

  model.add(hugectr.DenseLayer(layer_type=hugectr.Layer_t.WeightMultiply,
                               bottom_names=["slice21"],
                               top_names=["weight_multiply1"],
                               weight_dims=[13, 10]))

  model.add(hugectr.DenseLayer(layer_type=hugectr.Layer_t.WeightMultiply,
                               bottom_names=["slice22"],
                               top_names=["weight_multiply2"],
                               weight_dims=[13, 1]))

  model.add(hugectr.DenseLayer(layer_type=hugectr.Layer_t.Concat,
                               bottom_names=["reshape2", "weight_multiply1"],
                               top_names=["concat1"]))

  model.add(hugectr.DenseLayer(layer_type=hugectr.Layer_t.Slice,
                               bottom_names=["concat1"],
                               top_names=["slice31", "slice32"],
                               ranges=[(0, 390), (0, 390)]))

  model.add(hugectr.DenseLayer(layer_type=hugectr.Layer_t.InnerProduct,
                               bottom_names=["slice31"],
                               top_names=["fc1"],
                               num_output=400))

  model.add(hugectr.DenseLayer(layer_type=hugectr.Layer_t.ReLU,
                               bottom_names=["fc1"],
                               top_names=["relu1"]))

  model.add(hugectr.DenseLayer(layer_type=hugectr.Layer_t.Dropout,
                               bottom_names=["relu1"],
                               top_names=["dropout1"],
                               dropout_rate=dropout_rate))

  model.add(hugectr.DenseLayer(layer_type=hugectr.Layer_t.InnerProduct,
                               bottom_names=["dropout1"],
                               top_names=["fc2"],
                               num_output=400))

  model.add(hugectr.DenseLayer(layer_type=hugectr.Layer_t.ReLU,
                               bottom_names=["fc2"],
                               top_names=["relu2"]))

  model.add(hugectr.DenseLayer(layer_type=hugectr.Layer_t.Dropout,
                               bottom_names=["relu2"],
                               top_names=["dropout2"],
                               dropout_rate=dropout_rate))

  model.add(hugectr.DenseLayer(layer_type=hugectr.Layer_t.InnerProduct,
                               bottom_names=["dropout2"],
                               top_names=["fc3"],
                               num_output=400))

  model.add(hugectr.DenseLayer(layer_type=hugectr.Layer_t.ReLU,
                               bottom_names=["fc3"],
                               top_names=["relu3"]))

  model.add(hugectr.DenseLayer(layer_type=hugectr.Layer_t.Dropout,
                               bottom_names=["relu3"],
                               top_names=["dropout3"],
                               dropout_rate=dropout_rate))
  model.add(hugectr.DenseLayer(layer_type=hugectr.Layer_t.InnerProduct,
                               bottom_names=["dropout3"],
                               top_names=["fc4"],
                               num_output=1))

  model.add(hugectr.DenseLayer(layer_type=hugectr.Layer_t.FmOrder2,
                               bottom_names=["slice32"],
                               top_names=["fmorder2"],
                               out_dim=10))

  model.add(hugectr.DenseLayer(layer_type=hugectr.Layer_t.ReduceSum,
                               bottom_names=["fmorder2"],
                               top_names=["reducesum1"],
                               axis=1))

  model.add(hugectr.DenseLayer(layer_type=hugectr.Layer_t.Concat,
                               bottom_names=["reshape3", "weight_multiply2"],
                               top_names=["concat2"]))

  model.add(hugectr.DenseLayer(layer_type=hugectr.Layer_t.ReduceSum,
                               bottom_names=["concat2"],
                               top_names=["reducesum2"],
                               axis=1))

  model.add(hugectr.DenseLayer(
      layer_type=hugectr.Layer_t.Add,
      bottom_names=["fc4", "reducesum1", "reducesum2"],
      top_names=["add"]))

  model.add(hugectr.DenseLayer(
      layer_type=hugectr.Layer_t.BinaryCrossEntropyLoss,
      bottom_names=["add", "label"],
      top_names=["loss"]))

  model.compile()

  return model

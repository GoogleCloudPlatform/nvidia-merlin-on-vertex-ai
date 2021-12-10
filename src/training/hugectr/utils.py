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
"""Utilities for model training."""

from nvtabular.graph.schema import Schema
from nvtabular.graph.tags import Tags


def retrieve_cardinalities(schema_path):
  """Retrieves cardinalities from schema."""

  schema = Schema.load(schema_path)
  cardinalities = {
      key: value.properties['embedding_sizes']['cardinality']
      for key, value in schema.column_schemas.items()
      if Tags.CATEGORICAL in value.tags
  }

  return cardinalities

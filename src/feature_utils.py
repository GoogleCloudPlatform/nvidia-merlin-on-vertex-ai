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
"""Dataset features."""
from typing import Dict, Union
import numpy as np


def get_criteo_col_dtypes() -> Dict[str, Union[str, np.int32]]:
  """Returns a dict mapping column names to numpy dtype."""
  # Specify column dtypes. Note that "hex" means that
  # the values will be hexadecimal strings that should
  # be converted to int32
  col_dtypes = {}

  col_dtypes["label"] = np.int32
  for x in ["I" + str(i) for i in range(1, 14)]:
    col_dtypes[x] = np.int32
  for x in ["C" + str(i) for i in range(1, 27)]:
    col_dtypes[x] = "hex"

  return col_dtypes


def categorical_columns():
  return ["C" + str(x) for x in range(1, 27)]


def continuous_columns():
  return ["I" + str(x) for x in range(1, 14)]


def label_columns():
  return ["label"]

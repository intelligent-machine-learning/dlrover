# Copyright 2023 The DLRover Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from deepfm import DeepFM

from dlrover.trainer.tensorflow.util.column_info import Column


class TrainConf(object):
    classifier_class = DeepFM
    epoch = 10000
    batch_size = 2
    log_steps = 10
    model_dir = "/nas"
    enable_incr_saved_model = True
    checkpoint_incremental_save_secs = 800

    sparse_features = ["C" + str(i) for i in range(1, 27)]
    dense_features = ["I" + str(i) for i in range(1, 14)]

    params = {
        "dense_features": dense_features,
        "sparse_features": sparse_features,
    }

    dense_col = [
        Column.create(  # type: ignore
            name="I" + str(i),
            dtype="float32" if i != 0 else "int32",
            is_sparse=False,
            is_label=False if i != 0 else True,
        )
        for i in range(0, 14)
    ]

    sparse_col = [
        Column.create(  # type: ignore
            name="C" + str(i), dtype="string", is_sparse=True, is_label=False
        )
        for i in range(1, 27)
    ]

    col = dense_col + sparse_col
    train_set = {
        "path": "file://./data_kaggle_ad_ctr_train.csv",
        "columns": col,
    }

    eval_set = {
        "path": "file://./data_kaggle_ad_ctr_train.csv",
        "columns": train_set["columns"],
    }

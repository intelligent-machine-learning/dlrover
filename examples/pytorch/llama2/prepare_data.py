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

import json
import os

import fire
import pandas as pd


def sentiment_score_to_name(score: float):
    if score > 0:
        return "Positive"
    elif score < 0:
        return "Negative"
    return "Neutral"


def convert_dataset(src_data_path, output_dir=""):
    df = pd.read_csv(src_data_path)
    dataset_data = [
        {
            "instruction": "Detect the sentiment of the tweet.",
            "input": row_dict["Tweet"],
            "output": sentiment_score_to_name(row_dict["sent_score"]),
        }
        for row_dict in df.to_dict(orient="records")
    ]
    sample_count = df.shape[0]
    print(f"The dataset has {sample_count} samples.")

    if not output_dir:
        output_dir = os.path.dirname(src_data_path)
    elif not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, "btc_tweets_sentiment.json")
    with open(output_path, "w") as f:
        json.dump(dataset_data, f)


if __name__ == "__main__":
    fire.Fire(convert_dataset)

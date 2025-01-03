# Copyright 2024 The DLRover Authors. All rights reserved.
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

from datetime import datetime, timedelta


def has_expired(timestamp: int, time_period: int) -> bool:
    """
    Check if a give timestamp has expired

    Args:
        timestamp: data timestamp in seconds
        time_period: expire time period in seconds
    Return:
        indicate if the timestamp has expired
    """
    dt = datetime.fromtimestamp(timestamp)
    expired_dt = dt + timedelta(seconds=time_period)
    return expired_dt < datetime.now()


def timestamp_diff_in_seconds(timestamp1: float, timestamp2: float) -> float:
    dt1 = datetime.fromtimestamp(timestamp1)
    dt2 = datetime.fromtimestamp(timestamp2)
    if timestamp1 == timestamp2:
        return 0
    elif timestamp1 > timestamp2:
        return (dt1 - dt2).total_seconds()
    else:
        return (dt2 - dt1).total_seconds()

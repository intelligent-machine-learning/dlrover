// Copyright 2022 The DLRover Authors. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package utils

import (
	"fmt"
	"github.com/intelligent-machine-learning/easydl/brain/pkg/common"
	datastoreapi "github.com/intelligent-machine-learning/easydl/brain/pkg/datastore/api"
	"github.com/intelligent-machine-learning/easydl/brain/pkg/datastore/recorder/mysql"
)

// GetData returns data for a given condition
func GetData(client *mysql.Client, condition *datastoreapi.Condition, data interface{}) error {
	switch condition.Type {
	case common.TypeGetDataGetJobMetrics:
		_, ok := condition.Extra.(*mysql.JobMetricsCondition)
		if !ok {
			return fmt.Errorf("GetData %s condition is not *JobMetricsCondition", condition.Type)
		}
		_, ok = data.(*mysql.JobMetrics)
		if !ok {
			return fmt.Errorf("GetData %s data is not *JobMetrics", condition.Type)
		}
		return client.JobMetricsRecorder.Get(condition.Extra.(*mysql.JobMetricsCondition), data.(*mysql.JobMetrics))
	case common.TypeGetDataListJobMetrics:
		_, ok := condition.Extra.(*mysql.JobMetricsCondition)
		if !ok {
			return fmt.Errorf("GetData %s condition is not *JobMetricsCondition", condition.Type)
		}
		_, ok = data.(*[]*mysql.JobMetrics)
		if !ok {
			return fmt.Errorf("GetData %s data is not *[]*JobMetrics", condition.Type)
		}
		return client.JobMetricsRecorder.List(condition.Extra.(*mysql.JobMetricsCondition), data.(*[]*mysql.JobMetrics))
	}
	return fmt.Errorf("invalid type: %s", condition.Type)
}

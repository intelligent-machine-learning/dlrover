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

package optalgorithm

import (
	"encoding/json"
	"github.com/intelligent-machine-learning/easydl/brain/pkg/common"
	"github.com/intelligent-machine-learning/easydl/brain/pkg/config"
	dsimpl "github.com/intelligent-machine-learning/easydl/brain/pkg/datastore/implementation"
	"github.com/intelligent-machine-learning/easydl/brain/pkg/datastore/recorder/mysql"
	optconfig "github.com/intelligent-machine-learning/easydl/brain/pkg/optimizer/config"
	optimplcomm "github.com/intelligent-machine-learning/easydl/brain/pkg/optimizer/implementation/common"

	"github.com/stretchr/testify/assert"
	"testing"
)

func TestOptimizeJobWorkerCreateResource(t *testing.T) {
	dataStore := &dsimpl.BaseDataStore{
		Client: mysql.NewFakeClient(),
	}

	maxMem := 14.0 * 1024 * 1024 * 1024
	maxCPU := 14.0
	rt1 := []*common.JobRuntimeInfo{
		{
			WorkerCPU: map[uint64]float64{
				0: 11,
				1: 12,
			},
			WorkerMemory: map[uint64]float64{
				0: maxMem,
				1: 2 * 1024 * 1024 * 1024,
			},
		},
	}

	rt2 := []*common.JobRuntimeInfo{
		{
			WorkerCPU: map[uint64]float64{
				0: maxCPU,
				1: 11,
			},
			WorkerMemory: map[uint64]float64{
				0: 1 * 1024 * 1024 * 1024,
				1: 2 * 1024 * 1024 * 1024,
			},
		},
	}

	rtStr1, err := json.Marshal(rt1)
	assert.NoError(t, err)
	rtStr2, err := json.Marshal(rt2)
	assert.NoError(t, err)

	historyJobs := []*common.OptimizeJobMeta{
		{
			Metrics: &common.JobMetrics{
				JobUUID:    "uuid-1",
				JobName:    "job-1",
				JobRuntime: string(rtStr1),
				ExitReason: optimplcomm.ExitReasonCompleted,
			},
		},
		{
			Metrics: &common.JobMetrics{
				JobUUID:    "uuid-2",
				JobName:    "job-2",
				JobRuntime: string(rtStr2),
				ExitReason: optimplcomm.ExitReasonCompleted,
			},
		},
	}

	///////////////////////////////////////
	conf := &optconfig.OptimizeAlgorithmConfig{
		CustomizedConfig: map[string]string{
			config.JobNodeMemoryMarginPercent: "0.4",
		},
	}
	plan, err := OptimizeJobWorkerCreateResource(dataStore, conf, nil, historyJobs)
	assert.NoError(t, err)
	assert.NotEqual(t, plan, nil)
	res := plan.JobRes.TaskGroupResources[common.WorkerTaskGroupName]

	configMemory := maxMem * 1.4
	configCPU := float32(maxCPU)
	assert.Equal(t, res.Resource.Memory, configMemory)
	assert.Equal(t, res.Resource.CPUCore, configCPU)
}

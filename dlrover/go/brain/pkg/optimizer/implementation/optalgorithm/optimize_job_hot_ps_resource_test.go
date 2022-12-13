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
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestOptimizeJobHotPSResource(t *testing.T) {
	conf := &optconfig.OptimizeAlgorithmConfig{
		CustomizedConfig: map[string]string{
			config.OptimizerHotPSCPUThreshold:         "0.8",
			config.OptimizerHotPSMemoryThreshold:      "0.9",
			config.OptimizerHotPSCPUTargetWorkerCount: "20",
			config.OptimizerHotPSMemoryAdjust:         "4000000000",
		},
	}

	jobUUID := "uuid"
	///////////////////////////////////////////////////////////
	cpu1 := 6
	cpu2 := 9
	memory := 4 * 1024 * 1024 * 1024
	runtimes := []*common.JobRuntimeInfo{
		{
			PSCPU: map[uint64]float64{
				0: float64(cpu1),
				1: float64(cpu2),
			},
			PSMemory: map[uint64]float64{
				0: float64(memory),
				1: float64(memory),
			},
		},
		{
			PSCPU: map[uint64]float64{
				0: float64(cpu1),
				1: float64(cpu2),
			},
			PSMemory: map[uint64]float64{
				0: float64(memory),
				1: float64(memory),
			},
		},
		{
			PSCPU: map[uint64]float64{
				0: float64(cpu1),
				1: float64(cpu2),
			},
			PSMemory: map[uint64]float64{
				0: float64(memory),
				1: float64(memory),
			},
		},
	}
	runtimeStr, err := json.Marshal(runtimes)
	assert.NoError(t, err)

	optJob := &common.OptimizeJobMeta{
		JobMeta: &common.JobMeta{
			Name: "job",
			UUID: jobUUID,
		},
		Metrics: &common.JobMetrics{
			JobRuntime: string(runtimeStr),
		},
	}

	////////////////////////////////////////////////////////////////
	dataStore := &dsimpl.BaseDataStore{
		Client: mysql.NewFakeClient(),
	}

	nodeName1 := "task-ps-0"
	nodeName2 := "task-ps-1"

	res1 := &common.PodResource{
		CPUCore: 10,
		Memory:  5 * 1024 * 1024 * 1024,
	}
	resStr1, err := json.Marshal(res1)
	assert.NoError(t, err)
	node1 := &mysql.JobNode{
		Name:     nodeName1,
		UID:      "task-uid-0",
		JobUUID:  jobUUID,
		Type:     common.PSTaskGroupName,
		Resource: string(resStr1),
	}
	dataStore.Client.JobNodeRecorder.Upsert(node1)

	res2 := &common.PodResource{
		CPUCore: 10,
		Memory:  5 * 1024 * 1024 * 1024,
	}
	resStr2, err := json.Marshal(res2)
	assert.NoError(t, err)

	node2 := &mysql.JobNode{
		Name:     nodeName2,
		UID:      "task-uid-1",
		JobUUID:  jobUUID,
		Type:     common.PSTaskGroupName,
		Resource: string(resStr2),
	}
	dataStore.Client.JobNodeRecorder.Upsert(node2)

	/////////////////////////////////////////////////////////////////////////////
	plan, err := OptimizeJobHotPSResource(dataStore, conf, optJob, nil)
	assert.NoError(t, err)
	assert.NotEqual(t, plan, (*common.AlgorithmOptimizePlan)(nil))
	res := plan.JobRes.PodResources[nodeName2]
	assert.Equal(t, res.CPUCore, float32(32))
}

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

func TestOptimizeJobPSResourceUtil(t *testing.T) {
	conf := &optconfig.OptimizeAlgorithmConfig{
		CustomizedConfig: map[string]string{
			config.OptimizerHotPSCPUTargetWorkerCount: "3",
			config.OptimizerLowPSCPUThreshold:         "0.4",
			config.OptimizerPSMemoryMarginPercent:     "0.5",
			config.OptimizerPSMarginCPU:               "4",
			config.OptimizerPSCPUOverload:             "0.1",
		},
	}

	jobUUID := "uuid"
	///////////////////////////////////////////////////////////
	cpu1 := 2
	cpu2 := 4
	memory := 4 * 1024 * 1024 * 1024
	runtimes := []*common.JobRuntimeInfo{
		{
			GlobalStep: 1,
			Speed:      2,
			PSCPU: map[uint64]float64{
				0: float64(cpu1),
			},
			PSMemory: map[uint64]float64{
				0: float64(memory),
			},
			WorkerCPU: map[uint64]float64{
				0: float64(cpu1),
				1: float64(cpu1),
				2: float64(cpu1),
			},
		},
		{
			GlobalStep: 10,
			Speed:      2,
			PSCPU: map[uint64]float64{
				0: float64(cpu2),
			},
			PSMemory: map[uint64]float64{
				0: float64(memory),
			},
			WorkerCPU: map[uint64]float64{
				0: float64(cpu1),
				1: float64(cpu1),
				2: float64(cpu1),
			},
		},
		{
			GlobalStep: 20,
			Speed:      2,
			PSCPU: map[uint64]float64{
				0: float64(cpu1),
			},
			PSMemory: map[uint64]float64{
				0: float64(memory),
				1: float64(memory),
			},
			WorkerCPU: map[uint64]float64{
				0: float64(cpu1),
				1: float64(cpu1),
				2: float64(cpu1),
			},
		},
	}
	runtimeStr, err := json.Marshal(runtimes)
	assert.NoError(t, err)

	hyperParam := &common.TrainingHyperParams{
		BatchSize: 10,
	}
	hyperParamStr, err := json.Marshal(hyperParam)
	assert.NoError(t, err)

	setFeature := &common.TrainingSetFeature{
		DatasetSize: 200000,
	}
	setFeatureStr, err := json.Marshal(setFeature)
	assert.NoError(t, err)

	optJob := &common.OptimizeJobMeta{
		JobMeta: &common.JobMeta{
			Name: "job",
			UUID: jobUUID,
		},
		Metrics: &common.JobMetrics{
			JobRuntime:         string(runtimeStr),
			DatasetFeature:     string(setFeatureStr),
			HyperParamsFeature: string(hyperParamStr),
		},
	}

	////////////////////////////////////////////////////////////////
	dataStore := &dsimpl.BaseDataStore{
		Client: mysql.NewFakeClient(),
	}

	nodeName1 := "task-ps-0"

	nodeRes := &common.PodResource{
		CPUCore: 20,
		Memory:  5 * 1024 * 1024 * 1024,
	}
	nodeResStr, _ := json.Marshal(nodeRes)

	node1 := &mysql.JobNode{
		Name:     nodeName1,
		UID:      "task-uid-1",
		JobUUID:  jobUUID,
		Type:     common.PSTaskGroupName,
		Resource: string(nodeResStr),
	}
	dataStore.Client.JobNodeRecorder.Upsert(node1)

	/////////////////////////////////////////////////////////////////////////////
	plan, err := OptimizeJobPSResourceUtil(dataStore, conf, optJob, nil)
	assert.NoError(t, err)
	assert.NotEqual(t, plan, (*common.AlgorithmOptimizePlan)(nil))
	res := plan.JobRes.PodResources[nodeName1]
	assert.Equal(t, res.CPUCore, float32(8))
}

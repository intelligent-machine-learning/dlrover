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

func TestOptimizeJobPSOomResource1(t *testing.T) {
	dataStore := &dsimpl.BaseDataStore{
		Client: mysql.NewFakeClient(),
	}

	jobUUID := "job-uuid-1"
	jobName := "job-name-1"

	job := &common.OptimizeJobMeta{
		JobMeta: &common.JobMeta{
			Name: jobName,
			UUID: jobUUID,
		},
		Metrics: &common.JobMetrics{
			JobRuntime: "",
		},
	}

	/////////////////////////////////////
	nodeRes1 := &common.PodResource{
		Memory: 110,
	}
	nodeResStr1, err := json.Marshal(nodeRes1)
	assert.NoError(t, err)

	nodeStatus1 := &common.JobNodeStatus{
		Status: "Running",
		IsOOM:  false,
	}
	nodeStatusStr1, err := json.Marshal(nodeStatus1)
	assert.NoError(t, err)

	node1 := &mysql.JobNode{
		UID:      "task-uuid-1",
		JobUUID:  jobUUID,
		Type:     common.PSTaskGroupName,
		Status:   string(nodeStatusStr1),
		Resource: string(nodeResStr1),
	}
	dataStore.Client.JobNodeRecorder.Upsert(node1)

	nodeRes2 := &common.PodResource{
		Memory: 100,
	}
	nodeResStr2, err := json.Marshal(nodeRes2)
	assert.NoError(t, err)

	nodeStatus2 := &common.JobNodeStatus{
		Status: "Error",
		IsOOM:  true,
	}
	nodeStatusStr2, err := json.Marshal(nodeStatus2)
	assert.NoError(t, err)

	node2 := &mysql.JobNode{
		UID:      "task-uuid-2",
		JobUUID:  jobUUID,
		Type:     common.PSTaskGroupName,
		Status:   string(nodeStatusStr2),
		Resource: string(nodeResStr2),
	}
	dataStore.Client.JobNodeRecorder.Upsert(node2)

	///////////////////////////////////////
	conf := &optconfig.OptimizeAlgorithmConfig{
		CustomizedConfig: map[string]string{
			config.OptimizerPSMemoryWorkloadUnbalancePercent: "0.2",
		},
	}
	plan, err := OptimizeJobPSOomResource(dataStore, conf, job, nil)
	assert.NoError(t, err)
	assert.NotEqual(t, plan, nil)
	res := plan.JobRes.TaskGroupResources[common.PSTaskGroupName]

	optMemory := 110 * 2.0
	optReplica := 0
	assert.Equal(t, res.Resource.Memory, optMemory)
	assert.Equal(t, res.Count, int32(optReplica))
}

func TestOptimizeJobPSOomResource2(t *testing.T) {
	dataStore := &dsimpl.BaseDataStore{
		Client: mysql.NewFakeClient(),
	}

	jobUUID := "job-uuid-1"
	jobName := "job-name-1"

	rts := []*common.JobRuntimeInfo{
		{
			PSMemory: map[uint64]float64{
				0: 100,
				1: 100,
			},
		},
	}

	rtsStr, err := json.Marshal(rts)
	assert.NoError(t, err)

	job := &common.OptimizeJobMeta{
		JobMeta: &common.JobMeta{
			Name: jobName,
			UUID: jobUUID,
		},
		Metrics: &common.JobMetrics{
			JobRuntime: string(rtsStr),
		},
	}

	///////////////////////////////////////
	conf := &optconfig.OptimizeAlgorithmConfig{
		CustomizedConfig: map[string]string{
			config.OptimizerPSMemoryWorkloadUnbalancePercent: "0.2",
		},
	}
	plan, err := OptimizeJobPSOomResource(dataStore, conf, job, nil)
	assert.NoError(t, err)
	assert.NotEqual(t, plan, nil)
	res := plan.JobRes.TaskGroupResources[common.PSTaskGroupName]

	optMemory := 0.0
	optReplica := 4
	assert.Equal(t, res.Resource.Memory, optMemory)
	assert.Equal(t, res.Count, int32(optReplica))
}

func TestOptimizeJobPSOomResource3(t *testing.T) {
	dataStore := &dsimpl.BaseDataStore{
		Client: mysql.NewFakeClient(),
	}

	jobUUID := "job-uuid-1"
	jobName := "job-name-1"

	rts := []*common.JobRuntimeInfo{
		{
			PSMemory: map[uint64]float64{
				0: 100,
				1: 200,
			},
		},
	}

	rtsStr, err := json.Marshal(rts)
	assert.NoError(t, err)

	job := &common.OptimizeJobMeta{
		JobMeta: &common.JobMeta{
			Name: jobName,
			UUID: jobUUID,
		},
		Metrics: &common.JobMetrics{
			JobRuntime: string(rtsStr),
		},
	}

	///////////////////////////////////////
	conf := &optconfig.OptimizeAlgorithmConfig{
		CustomizedConfig: map[string]string{
			config.OptimizerPSMemoryWorkloadUnbalancePercent: "0.2",
		},
	}
	plan, err := OptimizeJobPSOomResource(dataStore, conf, job, nil)
	assert.NoError(t, err)
	assert.NotEqual(t, plan, nil)
	res := plan.JobRes.TaskGroupResources[common.PSTaskGroupName]

	optMemory := 200 * 2.0
	optReplica := 0
	assert.Equal(t, res.Resource.Memory, optMemory)
	assert.Equal(t, res.Count, int32(optReplica))
}

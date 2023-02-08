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

package optimizer

import (
	"encoding/json"
	"github.com/intelligent-machine-learning/easydl/brain/pkg/common"
	"github.com/intelligent-machine-learning/easydl/brain/pkg/config"
	dsimpl "github.com/intelligent-machine-learning/easydl/brain/pkg/datastore/implementation"
	"github.com/intelligent-machine-learning/easydl/brain/pkg/datastore/recorder/mysql"
	optconfig "github.com/intelligent-machine-learning/easydl/brain/pkg/optimizer/config"
	"github.com/stretchr/testify/assert"
	"math"
	"testing"
)

func TestJobPSInitialResourceOptimizer(t *testing.T) {
	dataStore := &dsimpl.BaseDataStore{
		Client: mysql.NewFakeClient(),
	}

	////////////////////////////////////////////////////////////
	jobUUID := "uuid"
	jobName := "job"

	jobMetas := []*common.JobMeta{
		{
			UUID: jobUUID,
			Name: jobName,
		},
	}

	////////////////////////////////////////////////////////////
	job := &mysql.Job{
		Name:   jobName,
		UID:    jobUUID,
		Status: "",
	}
	dataStore.Client.JobRecorder.Upsert(job)

	///////////////////////////////////////////////////////////
	cpu1 := 6
	cpu2 := 8
	recvOpCount := 249

	runtimes := []*common.JobRuntimeInfo{
		{
			PSCPU: map[uint64]float64{
				0: float64(cpu1),
				1: 10,
			},
		},
		{
			PSCPU: map[uint64]float64{
				0: 10,
				1: float64(cpu2),
			},
		},
	}
	runtimeStr, err := json.Marshal(runtimes)
	assert.NoError(t, err)

	modelFeature := &common.ModelFeature{
		RecvOpCount: uint64(recvOpCount),
	}
	modelFeatureStr, err := json.Marshal(modelFeature)
	assert.NoError(t, err)

	jobMetrics := &mysql.JobMetrics{
		UID:          jobUUID,
		JobRuntime:   string(runtimeStr),
		ModelFeature: string(modelFeatureStr),
	}
	dataStore.Client.JobMetricsRecorder.Upsert(jobMetrics)

	////////////////////////////////////////////////////////////////////
	optimizerConf := &optconfig.OptimizerConfig{
		OptimizeAlgorithmConfig: &optconfig.OptimizeAlgorithmConfig{},
	}

	conf := config.NewEmptyConfig()
	testOptimizer := newJobPSInitAdjustResourceOptimizer(dataStore, conf)
	plans, err := testOptimizer.Optimize(optimizerConf, jobMetas)

	optCPU := math.Ceil(0.08*float64(recvOpCount)/2 + 4)

	assert.NoError(t, err)
	assert.Equal(t, len(plans), 1)
	res := plans[0].AlgOptPlan.JobRes.TaskGroupResources[common.PSTaskGroupName]
	assert.Equal(t, res.Resource.CPUCore, float32(optCPU))
}

func TestJobPSResourceOptimizerForPSOom(t *testing.T) {
	dataStore := &dsimpl.BaseDataStore{
		Client: mysql.NewFakeClient(),
	}

	////////////////////////////////////////////////////////////
	jobUUID := "uuid"
	jobName := "job"

	jobMetas := []*common.JobMeta{
		{
			UUID: jobUUID,
			Name: jobName,
		},
	}

	////////////////////////////////////////////////////////////
	status := &common.JobStatus{
		IsOOM: true,
	}
	statusStr, err := json.Marshal(status)
	assert.NoError(t, err)

	job := &mysql.Job{
		Name:   jobName,
		UID:    jobUUID,
		Status: string(statusStr),
	}
	dataStore.Client.JobRecorder.Upsert(job)

	//////////////////////////////////////////////////////////////
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

	jobMetrics := &mysql.JobMetrics{
		UID:        jobUUID,
		JobRuntime: string(rtsStr),
	}
	dataStore.Client.JobMetricsRecorder.Upsert(jobMetrics)

	//////////////////////////////////////////////////////
	conf := config.NewEmptyConfig()
	testOptimizer := newJobPSInitAdjustResourceOptimizer(dataStore, conf)

	optimizerConf := &optconfig.OptimizerConfig{
		OptimizeAlgorithmConfig: &optconfig.OptimizeAlgorithmConfig{},
	}
	plans, err := testOptimizer.Optimize(optimizerConf, jobMetas)

	assert.NoError(t, err)
	assert.Equal(t, len(plans), 1)
	res := plans[0].AlgOptPlan.JobRes.TaskGroupResources[common.PSTaskGroupName]

	optMemory := 0.0
	optReplica := 2 * 2
	assert.Equal(t, res.Resource.Memory, optMemory)
	assert.Equal(t, res.Count, int32(optReplica))
}

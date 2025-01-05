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
	optimplcomm "github.com/intelligent-machine-learning/easydl/brain/pkg/optimizer/implementation/common"
	"github.com/stretchr/testify/assert"
	"math"
	"testing"
	"time"
)

func TestOptimizeJobPSCreateResource(t *testing.T) {
	dataStore := &dsimpl.BaseDataStore{
		Client: mysql.NewFakeClient(),
	}

	jobUUID := "job-uuid-0"
	jobName := "job-name-0"

	jobUUID1 := "job-uuid-1"
	jobName1 := "job-name-1"

	jobUUID2 := "job-uuid-2"
	jobName2 := "job-name-2"

	scenario := "user"

	jobMetas := []*common.JobMeta{
		{
			UUID: jobUUID,
			Name: jobName,
		},
	}

	/////////////////////////////////////////////////////////////////////////
	job := &mysql.Job{
		UID:      jobUUID,
		Name:     jobName,
		Scenario: scenario,
	}
	job1 := &mysql.Job{
		Name:      jobName1,
		UID:       jobUUID1,
		Scenario:  scenario,
		CreatedAt: time.Now().Add(time.Duration(-10) * time.Minute),
	}
	job2 := &mysql.Job{
		UID:       jobUUID2,
		Name:      jobName2,
		Scenario:  scenario,
		CreatedAt: time.Now().Add(time.Duration(-10) * time.Minute),
	}
	dataStore.Client.JobRecorder.Upsert(job)
	dataStore.Client.JobRecorder.Upsert(job1)
	dataStore.Client.JobRecorder.Upsert(job2)

	/////////////////////////////////////////////////////////////////////////
	maxCPU := 16

	cpu1 := 12
	cpu2 := 14
	memory1 := 1200
	memory2 := 1400

	runtimes1 := []*common.JobRuntimeInfo{
		{
			PSMemory: map[uint64]float64{
				0: 1000,
				1: 1200,
			},
			PSCPU: map[uint64]float64{
				0: float64(cpu1),
				1: 10,
			},
		},
		{
			PSMemory: map[uint64]float64{
				0: float64(memory1),
				1: float64(memory2),
			},
			PSCPU: map[uint64]float64{
				0: 10,
				1: float64(cpu2),
			},
		},
	}
	runtimeStr1, err := json.Marshal(runtimes1)
	assert.NoError(t, err)

	runtimes2 := []*common.JobRuntimeInfo{
		{
			PSMemory: map[uint64]float64{
				0: 500,
				1: 600,
			},
			PSCPU: map[uint64]float64{
				0: 6,
				1: float64(maxCPU),
			},
		},
		{
			PSMemory: map[uint64]float64{
				0: 600,
				1: 700,
			},
			PSCPU: map[uint64]float64{
				0: 5,
				1: float64(maxCPU),
			},
		},
	}
	runtimeStr2, err := json.Marshal(runtimes2)
	assert.NoError(t, err)

	jobMetrics := &mysql.JobMetrics{
		UID: jobUUID,
	}
	jobMetrics1 := &mysql.JobMetrics{
		UID:        jobUUID1,
		JobRuntime: string(runtimeStr1),
		ExitReason: optimplcomm.ExitReasonCompleted,
	}
	jobMetrics2 := &mysql.JobMetrics{
		UID:        jobUUID2,
		JobRuntime: string(runtimeStr2),
		ExitReason: optimplcomm.ExitReasonCompleted,
	}

	dataStore.Client.JobMetricsRecorder.Upsert(jobMetrics)
	dataStore.Client.JobMetricsRecorder.Upsert(jobMetrics1)
	dataStore.Client.JobMetricsRecorder.Upsert(jobMetrics2)

	/////////////////////////////////////////
	optimizerConf := &optconfig.OptimizerConfig{
		OptimizeAlgorithmConfig: &optconfig.OptimizeAlgorithmConfig{},
	}

	conf := config.NewEmptyConfig()
	conf.Set(config.QueryBackwardTimePeriodInHour, 5)
	testOptimizer := newJobPSCreateResourceOptimizer(dataStore, conf)
	plans, err := testOptimizer.Optimize(optimizerConf, jobMetas)
	assert.NoError(t, err)
	assert.Equal(t, len(plans), 1)
	res := plans[0].AlgOptPlan.JobRes.TaskGroupResources[common.PSTaskGroupName]

	optCPU := math.Ceil(float64(maxCPU)) + optimplcomm.DefaultCPUMargin
	totalCPU := float64(cpu1+cpu2) * 1.5
	replica := math.Ceil(totalCPU / optCPU)
	optMemory := float64(memory2) * (1 + optimplcomm.DefaultPSMemoryMarginPercent)

	assert.Equal(t, res.Count, int32(replica))
	assert.Equal(t, res.Resource.Memory, optMemory)
	assert.Equal(t, res.Resource.CPUCore, float32(optCPU))
}

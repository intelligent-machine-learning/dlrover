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
	"testing"
	"time"
)

func TestJobWorkerCreateResourceOptimizerWithoutOOM(t *testing.T) {
	dataStore := &dsimpl.BaseDataStore{
		Client: mysql.NewFakeClient(),
	}

	jobUUID := "job-uuid-0"
	jobName := "job-name-0"
	scenario := "scenario"

	jobMetas := []*common.JobMeta{
		{
			UUID: jobUUID,
			Name: jobName,
		},
	}

	job := &mysql.Job{
		Name:     jobName,
		UID:      jobUUID,
		Scenario: scenario,
	}
	dataStore.Client.JobRecorder.Upsert(job)

	jobMetrics := &mysql.JobMetrics{
		UID: jobUUID,
	}
	dataStore.Client.JobMetricsRecorder.Upsert(jobMetrics)

	///////////////////////////////////////////////////////////////////////
	jobUUID1 := "job-uuid-1"
	jobName1 := "job-name-1"

	jobUUID2 := "job-uuid-2"
	jobName2 := "job-name-2"

	job1 := &mysql.Job{
		Name:      jobName1,
		UID:       jobUUID1,
		Scenario:  scenario,
		CreatedAt: time.Now().Add(time.Duration(-10) * time.Minute),
	}
	job2 := &mysql.Job{
		Name:      jobName2,
		UID:       jobUUID2,
		Scenario:  scenario,
		CreatedAt: time.Now().Add(time.Duration(-10) * time.Minute),
	}
	dataStore.Client.JobRecorder.Upsert(job1)
	dataStore.Client.JobRecorder.Upsert(job2)

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

	jobMetrics1 := &mysql.JobMetrics{
		UID:        jobUUID1,
		JobRuntime: string(rtStr1),
		ExitReason: optimplcomm.ExitReasonCompleted,
	}
	jobMetrics2 := &mysql.JobMetrics{
		UID:        jobUUID2,
		JobRuntime: string(rtStr2),
		ExitReason: optimplcomm.ExitReasonCompleted,
	}
	dataStore.Client.JobMetricsRecorder.Upsert(jobMetrics1)
	dataStore.Client.JobMetricsRecorder.Upsert(jobMetrics2)

	/////////////////////////////////////////
	optConf := &optconfig.OptimizerConfig{}

	conf := config.NewEmptyConfig()
	testOptimizer := newJobWorkerCreateResourceOptimizer(dataStore, conf)
	plans, err := testOptimizer.Optimize(optConf, jobMetas)
	assert.NoError(t, err)
	assert.Equal(t, len(plans), 1)
	res := plans[0].AlgOptPlan.JobRes.TaskGroupResources[common.WorkerTaskGroupName]

	configMemory := maxMem * (1 + optimplcomm.DefaultMemoryMarginPercent)
	configCPU := maxCPU
	if configCPU < optimplcomm.DefaultWorkerCreateCPU {
		configCPU = optimplcomm.DefaultWorkerCreateCPU
	}
	assert.Equal(t, res.Resource.Memory, configMemory)
	assert.Equal(t, res.Resource.CPUCore, float32(configCPU))
}

func TestJobWorkerCreateResourceOptimizerWithOOM(t *testing.T) {
	dataStore := &dsimpl.BaseDataStore{
		Client: mysql.NewFakeClient(),
	}

	jobUUID := "job-uuid-0"
	jobName := "job-name-0"
	scenario := "scenario"

	jobMetas := []*common.JobMeta{
		{
			UUID: jobUUID,
			Name: jobName,
		},
	}

	jobStatus := &common.JobStatus{
		IsOOM: true,
	}
	jobStatusStr, err := json.Marshal(jobStatus)
	assert.NoError(t, err)
	job := &mysql.Job{
		Name:     jobName,
		UID:      jobUUID,
		Scenario: scenario,
		Status:   string(jobStatusStr),
	}
	dataStore.Client.JobRecorder.Upsert(job)

	jobMetrics := &mysql.JobMetrics{
		UID: jobUUID,
	}
	dataStore.Client.JobMetricsRecorder.Upsert(jobMetrics)

	///////////////////////////////////////////////////////////////////////
	jobUUID1 := "job-uuid-1"
	jobName1 := "job-name-1"

	job1 := &mysql.Job{
		Name:      jobName1,
		UID:       jobUUID1,
		Scenario:  scenario,
		CreatedAt: time.Now().Add(time.Duration(-10) * time.Minute),
	}
	dataStore.Client.JobRecorder.Upsert(job1)

	mem := 24000000000.0
	rt1 := []*common.JobRuntimeInfo{
		{
			WorkerMemory: map[uint64]float64{
				0: 24000000000,
				1: 10000000000,
				2: 26000000000,
			},
		},
		{
			WorkerMemory: map[uint64]float64{
				1: 10000000000,
				2: 26000000000,
			},
		},
	}
	rtStr1, err := json.Marshal(rt1)
	assert.NoError(t, err)

	jobMetrics1 := &mysql.JobMetrics{
		UID:        jobUUID1,
		JobRuntime: string(rtStr1),
		ExitReason: optimplcomm.ExitReasonCompleted,
	}
	dataStore.Client.JobMetricsRecorder.Upsert(jobMetrics1)

	////////////////////////////////////////////////////////////
	oomStatus := &common.JobNodeStatus{
		IsOOM: true,
	}
	oomStatusStr, err := json.Marshal(oomStatus)
	assert.NoError(t, err)

	nonOomStatus := &common.JobNodeStatus{
		IsOOM: false,
	}
	nonOomStatusStr, err := json.Marshal(nonOomStatus)
	assert.NoError(t, err)

	node1 := &mysql.JobNode{
		UID:     "node-1",
		Name:    "job-1-worker-0",
		JobUUID: jobUUID1,
		JobName: jobName1,
		Type:    common.WorkerTaskGroupName,
		Status:  string(oomStatusStr),
	}
	node2 := &mysql.JobNode{
		UID:     "node-2",
		Name:    "job-1-worker-1",
		JobUUID: jobUUID1,
		JobName: jobName1,
		Type:    common.WorkerTaskGroupName,
		Status:  string(nonOomStatusStr),
	}
	node3 := &mysql.JobNode{
		UID:     "node-3",
		Name:    "job-1-worker-2",
		JobUUID: jobUUID1,
		JobName: jobName1,
		Type:    common.WorkerTaskGroupName,
		Status:  string(nonOomStatusStr),
	}

	dataStore.Client.JobNodeRecorder.Upsert(node1)
	dataStore.Client.JobNodeRecorder.Upsert(node2)
	dataStore.Client.JobNodeRecorder.Upsert(node3)

	/////////////////////////////////////////
	optConf := &optconfig.OptimizerConfig{}

	conf := config.NewEmptyConfig()
	testOptimizer := newJobWorkerCreateResourceOptimizer(dataStore, conf)
	plans, err := testOptimizer.Optimize(optConf, jobMetas)
	assert.NoError(t, err)
	assert.Equal(t, len(plans), 1)
	res := plans[0].AlgOptPlan.JobRes.TaskGroupResources[common.WorkerTaskGroupName]

	configMem := mem * (1 + defaultOptimizerWorkerOomMemoryMarginPercent)
	if configMem < mem+defaultOptimizerWorkerOomMemoryMinIncrease {
		configMem = mem + defaultOptimizerWorkerOomMemoryMinIncrease
	}
	assert.Equal(t, res.Resource.Memory, configMem)
}

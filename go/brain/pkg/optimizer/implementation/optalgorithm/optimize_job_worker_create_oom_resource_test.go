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
	pb "github.com/intelligent-machine-learning/easydl/brain/pkg/proto"
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestOptimizeJobWorkerCreateOomResource1(t *testing.T) {
	dataStore := &dsimpl.BaseDataStore{
		Client: mysql.NewFakeClient(),
	}

	rt1 := []*common.JobRuntimeInfo{
		{
			WorkerMemory: map[uint64]float64{
				0: 13,
				1: 10,
				2: 14,
			},
		},
		{
			WorkerMemory: map[uint64]float64{
				1: 10,
				2: 14,
			},
		},
	}

	rtStr1, err := json.Marshal(rt1)
	assert.NoError(t, err)

	jobUUID := "job-uuid-1"
	jobName := "job-1"
	historyJobs := []*common.OptimizeJobMeta{
		{
			Metrics: &common.JobMetrics{
				JobUUID:    jobUUID,
				JobName:    jobName,
				JobRuntime: string(rtStr1),
				ExitReason: optimplcomm.ExitReasonCompleted,
			},
		},
	}

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
		JobUUID: jobUUID,
		JobName: jobName,
		Type:    common.WorkerTaskGroupName,
		Status:  string(oomStatusStr),
	}

	node2 := &mysql.JobNode{
		UID:     "node-2",
		Name:    "job-1-worker-1",
		JobUUID: jobUUID,
		JobName: jobName,
		Type:    common.WorkerTaskGroupName,
		Status:  string(nonOomStatusStr),
	}

	node3 := &mysql.JobNode{
		UID:     "node-3",
		Name:    "job-1-worker-2",
		JobUUID: jobUUID,
		JobName: jobName,
		Type:    common.WorkerTaskGroupName,
		Status:  string(nonOomStatusStr),
	}

	dataStore.Client.JobNodeRecorder.Upsert(node1)
	dataStore.Client.JobNodeRecorder.Upsert(node2)
	dataStore.Client.JobNodeRecorder.Upsert(node3)

	//////////////////////////////////////////////
	optJob := &common.OptimizeJobMeta{
		Metrics: &common.JobMetrics{},
		JobMeta: &common.JobMeta{
			Name: "job-0",
		},
	}

	///////////////////////////////////////
	conf := &optconfig.OptimizeAlgorithmConfig{
		CustomizedConfig: map[string]string{
			config.OptimizerWorkerOomMemoryMinIncrease:   "40",
			config.OptimizerWorkerOomMemoryMarginPercent: "0.2",
		},
	}
	plan, err := OptimizeJobWorkerCreateOomResource(dataStore, conf, optJob, historyJobs)
	assert.NoError(t, err)
	assert.NotEqual(t, plan, nil)
	res := plan.JobRes.TaskGroupResources[common.WorkerTaskGroupName]

	configMemory := 13 * 1.2
	assert.Equal(t, res.Resource.Memory, configMemory)
}

func TestOptimizeJobWorkerCreateOomResource2(t *testing.T) {
	dataStore := &dsimpl.BaseDataStore{
		Client: mysql.NewFakeClient(),
	}

	rt1 := []*common.JobRuntimeInfo{
		{
			WorkerMemory: map[uint64]float64{
				0: 13,
				1: 10,
				2: 14,
			},
		},
		{
			WorkerMemory: map[uint64]float64{
				1: 10,
				2: 14,
			},
		},
	}

	rtStr1, err := json.Marshal(rt1)
	assert.NoError(t, err)

	jobUUID := "job-uuid-1"
	jobName := "job-1"
	historyJobs := []*common.OptimizeJobMeta{
		{
			Metrics: &common.JobMetrics{
				JobUUID:    jobUUID,
				JobName:    jobName,
				JobRuntime: string(rtStr1),
				ExitReason: optimplcomm.ExitReasonCompleted,
			},
		},
	}

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
		JobUUID: jobUUID,
		JobName: jobName,
		Type:    common.WorkerTaskGroupName,
		Status:  string(oomStatusStr),
	}

	node2 := &mysql.JobNode{
		UID:     "node-2",
		Name:    "job-1-worker-1",
		JobUUID: jobUUID,
		JobName: jobName,
		Type:    common.WorkerTaskGroupName,
		Status:  string(nonOomStatusStr),
	}

	node3 := &mysql.JobNode{
		UID:     "node-3",
		Name:    "job-1-worker-2",
		JobUUID: jobUUID,
		JobName: jobName,
		Type:    common.WorkerTaskGroupName,
		Status:  string(nonOomStatusStr),
	}

	dataStore.Client.JobNodeRecorder.Upsert(node1)
	dataStore.Client.JobNodeRecorder.Upsert(node2)
	dataStore.Client.JobNodeRecorder.Upsert(node3)

	//////////////////////////////////////////////
	opt := []*pb.JobOptimization{
		{
			Plan: &pb.JobOptimizePlan{
				Resource: &pb.JobResource{
					TaskGroupResources: map[string]*pb.TaskGroupResource{
						common.WorkerTaskGroupName: {
							Resource: &pb.PodResource{
								Memory: 12,
							},
						},
					},
				},
			},
		},
	}
	optStr, err := json.Marshal(opt)
	assert.NoError(t, err)

	optJob := &common.OptimizeJobMeta{
		Metrics: &common.JobMetrics{
			Optimization: string(optStr),
		},
		JobMeta: &common.JobMeta{
			Name: "job-0",
		},
	}

	///////////////////////////////////////
	conf := &optconfig.OptimizeAlgorithmConfig{
		CustomizedConfig: map[string]string{
			config.OptimizerWorkerOomMemoryMinIncrease:   "4",
			config.OptimizerWorkerOomMemoryMarginPercent: "0.2",
		},
	}
	plan, err := OptimizeJobWorkerCreateOomResource(dataStore, conf, optJob, historyJobs)
	assert.NoError(t, err)
	assert.NotEqual(t, plan, nil)
	res := plan.JobRes.TaskGroupResources[common.WorkerTaskGroupName]

	configMemory := 16.0
	assert.Equal(t, res.Resource.Memory, configMemory)
}

func TestOptimizeJobWorkerCreateOomResource3(t *testing.T) {
	dataStore := &dsimpl.BaseDataStore{
		Client: mysql.NewFakeClient(),
	}

	rt1 := []*common.JobRuntimeInfo{
		{
			WorkerMemory: map[uint64]float64{
				0: 13,
				1: 10,
				2: 14,
			},
		},
		{
			WorkerMemory: map[uint64]float64{
				1: 10,
				2: 14,
			},
		},
	}

	rtStr1, err := json.Marshal(rt1)
	assert.NoError(t, err)

	jobUUID := "job-uuid-1"
	jobName := "job-1"
	historyJobs := []*common.OptimizeJobMeta{
		{
			Metrics: &common.JobMetrics{
				JobUUID:    jobUUID,
				JobName:    jobName,
				JobRuntime: string(rtStr1),
				ExitReason: optimplcomm.ExitReasonCompleted,
			},
		},
	}

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
		JobUUID: jobUUID,
		JobName: jobName,
		Type:    common.WorkerTaskGroupName,
		Status:  string(oomStatusStr),
	}

	node2 := &mysql.JobNode{
		UID:     "node-2",
		Name:    "job-1-worker-1",
		JobUUID: jobUUID,
		JobName: jobName,
		Type:    common.WorkerTaskGroupName,
		Status:  string(nonOomStatusStr),
	}

	node3 := &mysql.JobNode{
		UID:     "node-3",
		Name:    "job-1-worker-2",
		JobUUID: jobUUID,
		JobName: jobName,
		Type:    common.WorkerTaskGroupName,
		Status:  string(nonOomStatusStr),
	}

	dataStore.Client.JobNodeRecorder.Upsert(node1)
	dataStore.Client.JobNodeRecorder.Upsert(node2)
	dataStore.Client.JobNodeRecorder.Upsert(node3)

	//////////////////////////////////////////////
	rt2 := []*common.JobRuntimeInfo{
		{
			WorkerMemory: map[uint64]float64{
				1: 10,
				0: 12,
			},
		},
	}
	rtStr2, err := json.Marshal(rt2)
	assert.NoError(t, err)

	optJob := &common.OptimizeJobMeta{
		Metrics: &common.JobMetrics{
			JobRuntime: string(rtStr2),
		},
		JobMeta: &common.JobMeta{
			Name: "job-0",
		},
	}

	///////////////////////////////////////
	conf := &optconfig.OptimizeAlgorithmConfig{
		CustomizedConfig: map[string]string{
			config.OptimizerWorkerOomMemoryMinIncrease:   "4",
			config.OptimizerWorkerOomMemoryMarginPercent: "0.2",
		},
	}
	plan, err := OptimizeJobWorkerCreateOomResource(dataStore, conf, optJob, historyJobs)
	assert.NoError(t, err)
	assert.NotEqual(t, plan, nil)
	res := plan.JobRes.TaskGroupResources[common.WorkerTaskGroupName]

	configMemory := 16.0
	assert.Equal(t, res.Resource.Memory, configMemory)
}

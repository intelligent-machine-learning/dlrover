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
	"fmt"
	"github.com/intelligent-machine-learning/easydl/brain/pkg/common"
	"github.com/intelligent-machine-learning/easydl/brain/pkg/config"
	dsimpl "github.com/intelligent-machine-learning/easydl/brain/pkg/datastore/implementation"
	"github.com/intelligent-machine-learning/easydl/brain/pkg/datastore/recorder/mysql"
	optconfig "github.com/intelligent-machine-learning/easydl/brain/pkg/optimizer/config"
	"github.com/stretchr/testify/assert"
	"math"
	"strconv"
	"testing"
)

func TestOptimizeJobWorkerResource_AddReplica(t *testing.T) {
	dataStore := &dsimpl.BaseDataStore{
		Client: mysql.NewFakeClient(),
	}

	jobUUID := "job-uuid-1"
	cluster := "cluster"

	speed1 := float64(8)
	speed2 := float64(10)

	psCPU1 := 4
	psCPU2 := 6
	totalPSCPU := 10
	stepCount := 5
	psCPUOverload := 0.8

	maxMem := 20.0
	maxCPU := 0.35
	rtInfo1 := &common.JobRuntimeInfo{
		Speed: speed1,
		PSCPU: map[uint64]float64{
			0: float64(psCPU1),
		},
		WorkerCPU: map[uint64]float64{
			0: 0.3,
		},
		WorkerMemory: map[uint64]float64{
			0: 10,
		},
	}
	rtInfo2 := &common.JobRuntimeInfo{
		Speed: speed2,
		PSCPU: map[uint64]float64{
			0: float64(psCPU2),
		},
		WorkerCPU: map[uint64]float64{
			0: maxCPU,
			1: maxCPU,
			2: maxCPU,
			3: maxCPU,
			4: maxCPU,
		},
		WorkerMemory: map[uint64]float64{
			0: maxMem,
			1: maxMem,
			2: maxMem,
			3: maxMem,
			4: maxMem,
		},
	}

	rtInfos := make([]*common.JobRuntimeInfo, 0)
	for i := 0; i < stepCount; i++ {
		rtInfos = append(rtInfos, rtInfo1)
	}
	for i := 0; i < stepCount; i++ {
		rtInfos = append(rtInfos, rtInfo2)
	}

	rtInfosStr, err := json.Marshal(rtInfos)
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
			UUID:    jobUUID,
			Cluster: cluster,
		},
		Metrics: &common.JobMetrics{
			JobRuntime:         string(rtInfosStr),
			DatasetFeature:     string(setFeatureStr),
			HyperParamsFeature: string(hyperParamStr),
		},
	}

	conf := &optconfig.OptimizeAlgorithmConfig{
		CustomizedConfig: map[string]string{
			config.OptimizerWorkerMaxReplicaCount:      "10",
			config.OptimizerStepCountThreshold:         strconv.Itoa(stepCount),
			config.OptimizerPSCPUExhaustedThreshold:    "0.95",
			config.OptimizerPSCPUOverload:              fmt.Sprintf("%f", psCPUOverload),
			config.OptimizerTrainingSpeedLessPercent:   "0.1",
			config.OptimizerWorkerReplicaDecreaseCount: "1",
			config.OptimizerWorkerMaxInitCountPerStep:  "32",
			config.OptimizerWorkerMaxCountPerStep:      "4",
			config.OptimizerWorkerMemoryMarginPercent:  "0.2",
			config.OptimizerWorkerCPUMarginCore:        "1.0",
			config.OptimizerWorkerCPUUtilCompCount:     "2",
			config.OptimizerWorkerCPUUtilLessPercent:   "0.15",
			config.OptimizerWorkerOptimizePhase:        config.OptimizerWorkerOptimizePhaseStable,
		},
	}

	/////////////////////////////////////
	psRes := &common.PodResource{
		CPUCore: float32(totalPSCPU),
	}
	psResStr, _ := json.Marshal(psRes)

	psTask := &mysql.JobNode{
		Name:     "task-ps-0",
		UID:      "task-ps-uid-0",
		JobUUID:  jobUUID,
		Type:     common.PSTaskGroupName,
		Resource: string(psResStr),
	}

	dataStore.Client.JobNodeRecorder.Upsert(psTask)

	///////////////////////////////////////
	plan, err := OptimizeJobWorkerResource(dataStore, conf, optJob, nil)
	assert.NoError(t, err)
	assert.NotEqual(t, plan, nil)

	res := plan.JobRes.TaskGroupResources[common.WorkerTaskGroupName]
	replica := psCPUOverload / (float64(psCPU2) / float64(totalPSCPU)) * float64(len(rtInfo2.WorkerCPU))
	assert.Equal(t, res.Count, int32(math.Ceil(replica)))

	memory := maxMem * 1.2
	cpu := math.Ceil(maxCPU + 1.0)
	assert.Equal(t, res.Resource.Memory, memory)
	assert.Equal(t, res.Resource.CPUCore, float32(cpu))
}

func TestOptimizeJobWorkerResource_AddReplica2(t *testing.T) {
	dataStore := &dsimpl.BaseDataStore{
		Client: mysql.NewFakeClient(),
	}

	jobUUID := "job-uuid-1"
	cluster := "cluster"

	speed1 := float64(8)
	speed2 := float64(10)

	psCPU1 := 1
	psCPU2 := 2
	totalPSCPU := 10
	stepCount := 5
	psCPUOverload := 0.8

	rtInfo1 := &common.JobRuntimeInfo{
		Speed: speed1,
		PSCPU: map[uint64]float64{
			0: float64(psCPU1),
		},
		WorkerCPU: map[uint64]float64{
			0: 0.3,
		},
		WorkerMemory: map[uint64]float64{
			0: 10,
		},
	}
	rtInfo2 := &common.JobRuntimeInfo{
		Speed: speed2,
		PSCPU: map[uint64]float64{
			0: float64(psCPU2),
		},
		WorkerCPU: map[uint64]float64{
			0: 0.3,
		},
		WorkerMemory: map[uint64]float64{
			0: 10,
		},
	}

	rtInfos := make([]*common.JobRuntimeInfo, 0)
	for i := 0; i < stepCount; i++ {
		rtInfos = append(rtInfos, rtInfo1)
	}
	for i := 0; i < stepCount; i++ {
		rtInfos = append(rtInfos, rtInfo2)
	}

	rtInfosStr, err := json.Marshal(rtInfos)
	assert.NoError(t, err)

	hyperParam := &common.TrainingHyperParams{
		BatchSize: 10,
	}
	hyperParamStr, err := json.Marshal(hyperParam)
	assert.NoError(t, err)

	setFeature := &common.TrainingSetFeature{
		DatasetSize: 100000,
	}
	setFeatureStr, err := json.Marshal(setFeature)
	assert.NoError(t, err)

	optJob := &common.OptimizeJobMeta{
		JobMeta: &common.JobMeta{
			UUID:    jobUUID,
			Cluster: cluster,
		},
		Metrics: &common.JobMetrics{
			JobRuntime:         string(rtInfosStr),
			DatasetFeature:     string(setFeatureStr),
			HyperParamsFeature: string(hyperParamStr),
		},
	}

	conf := &optconfig.OptimizeAlgorithmConfig{
		CustomizedConfig: map[string]string{
			config.OptimizerWorkerMaxReplicaCount:      "10",
			config.OptimizerStepCountThreshold:         strconv.Itoa(stepCount),
			config.OptimizerPSCPUExhaustedThreshold:    "0.95",
			config.OptimizerPSCPUOverload:              fmt.Sprintf("%f", psCPUOverload),
			config.OptimizerTrainingSpeedLessPercent:   "0.1",
			config.OptimizerWorkerReplicaDecreaseCount: "1",
			config.OptimizerWorkerMaxInitCountPerStep:  "32",
			config.OptimizerWorkerMaxCountPerStep:      "4",
			config.OptimizerWorkerMemoryMarginPercent:  "0.2",
			config.OptimizerWorkerCPUMarginCore:        "1.0",
			config.OptimizerWorkerCPUUtilCompCount:     "2",
			config.OptimizerWorkerCPUUtilLessPercent:   "0.15",
			config.OptimizerWorkerOptimizePhase:        config.OptimizerWorkerOptimizePhaseSample,
		},
	}

	/////////////////////////////////////
	nodeRes := &common.PodResource{
		CPUCore: float32(totalPSCPU),
	}
	nodeResStr, _ := json.Marshal(nodeRes)

	node := &mysql.JobNode{
		Name:     "task-ps-0",
		UID:      "task-ps-uid-0",
		JobUUID:  jobUUID,
		Type:     common.PSTaskGroupName,
		Resource: string(nodeResStr),
	}
	dataStore.Client.JobNodeRecorder.Upsert(node)

	///////////////////////////////////////
	plan, err := OptimizeJobWorkerResource(dataStore, conf, optJob, nil)
	assert.NoError(t, err)
	assert.NotEqual(t, plan, nil)

	res := plan.JobRes.TaskGroupResources[common.WorkerTaskGroupName]
	replica := defaultInitWorker
	assert.Equal(t, res.Count, int32(replica))
}

func TestOptimizeJobWorkerResource_DecreaseReplica(t *testing.T) {
	dataStore := &dsimpl.BaseDataStore{
		Client: mysql.NewFakeClient(),
	}

	jobUUID := "job-uuid-1"
	cluster := "cluster"

	speed1 := float64(10)
	speed2 := float64(8)

	psCPU1 := 8
	psCPU2 := 9.6
	totalPSCPU := 10
	stepCount := 5
	psCPUOverload := 0.8

	rtInfo1 := &common.JobRuntimeInfo{
		Speed: speed1,
		PSCPU: map[uint64]float64{
			0: float64(psCPU1),
		},
		WorkerCPU: map[uint64]float64{
			0: 0.3,
		},
		WorkerMemory: map[uint64]float64{
			0: 10,
		},
	}
	rtInfo2 := &common.JobRuntimeInfo{
		Speed: speed2,
		PSCPU: map[uint64]float64{
			0: float64(psCPU2),
		},
		WorkerCPU: map[uint64]float64{
			0: 0.3,
			1: 0.35,
		},
		WorkerMemory: map[uint64]float64{
			0: 10,
			1: 10,
		},
	}

	rtInfos := make([]*common.JobRuntimeInfo, 0)
	for i := 0; i < stepCount; i++ {
		rtInfos = append(rtInfos, rtInfo1)
	}
	for i := 0; i < stepCount; i++ {
		rtInfos = append(rtInfos, rtInfo2)
	}

	rtInfosStr, err := json.Marshal(rtInfos)
	assert.NoError(t, err)

	optJob := &common.OptimizeJobMeta{
		JobMeta: &common.JobMeta{
			UUID:    jobUUID,
			Cluster: cluster,
		},
		Metrics: &common.JobMetrics{
			JobRuntime: string(rtInfosStr),
		},
	}

	conf := &optconfig.OptimizeAlgorithmConfig{
		CustomizedConfig: map[string]string{
			config.OptimizerWorkerMaxReplicaCount:      "10",
			config.OptimizerStepCountThreshold:         strconv.Itoa(stepCount),
			config.OptimizerPSCPUExhaustedThreshold:    "0.95",
			config.OptimizerPSCPUOverload:              fmt.Sprintf("%f", psCPUOverload),
			config.OptimizerTrainingSpeedLessPercent:   "0.1",
			config.OptimizerWorkerReplicaDecreaseCount: "1",
			config.OptimizerWorkerMaxInitCountPerStep:  "32",
			config.OptimizerWorkerMaxCountPerStep:      "4",
			config.OptimizerWorkerMemoryMarginPercent:  "0.2",
			config.OptimizerWorkerCPUMarginCore:        "1.0",
			config.OptimizerWorkerCPUUtilCompCount:     "2",
			config.OptimizerWorkerCPUUtilLessPercent:   "0.15",
		},
	}

	/////////////////////////////////////
	nodeRes := &common.PodResource{
		CPUCore: float32(totalPSCPU),
	}
	nodeResStr, _ := json.Marshal(nodeRes)

	node := &mysql.JobNode{
		Name:     "task-ps-0",
		UID:      "task-ps-uid-0",
		JobUUID:  jobUUID,
		Type:     common.PSTaskGroupName,
		Resource: string(nodeResStr),
	}
	dataStore.Client.JobNodeRecorder.Upsert(node)

	///////////////////////////////////////
	plan, err := OptimizeJobWorkerResource(dataStore, conf, optJob, nil)
	assert.NoError(t, err)
	assert.NotEqual(t, plan, nil)

	res := plan.JobRes.TaskGroupResources[common.WorkerTaskGroupName]
	replica := len(rtInfo2.WorkerCPU) - 1
	assert.Equal(t, res.Count, int32(replica))
}

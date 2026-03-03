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

func TestOptimizeJobPSInitialResource(t *testing.T) {
	dataStore := &dsimpl.BaseDataStore{
		Client: mysql.NewFakeClient(),
	}

	jobUUID := "job-uuid-1"
	cluster := "cluster"

	speed1 := float64(8)
	speed2 := float64(10)

	psCPU1 := 2
	psCPU2 := 3

	stepCount := 5
	psCPUOverload := 0.8
	psMemory1 := 4000
	psMemory2 := 10000

	rtInfo1 := &common.JobRuntimeInfo{
		Speed: speed1,
		PSCPU: map[uint64]float64{
			0: float64(psCPU1),
			1: float64(psCPU1),
		},
		PSMemory: map[uint64]float64{
			0: float64(psMemory1),
			1: float64(psMemory2),
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
			1: float64(psCPU2),
		},
		PSMemory: map[uint64]float64{
			0: float64(psMemory1),
			1: float64(psMemory2),
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

	recvOpCount := 249

	modelFeature := &common.ModelFeature{
		RecvOpCount: uint64(recvOpCount),
	}
	modelFeatureStr, err := json.Marshal(modelFeature)
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
			ModelFeature:       string(modelFeatureStr),
		},
	}

	conf := &optconfig.OptimizeAlgorithmConfig{
		CustomizedConfig: map[string]string{
			config.OptimizerStepCountThreshold:            strconv.Itoa(stepCount),
			config.OptimizerPSCPUOverload:                 fmt.Sprintf("%f", psCPUOverload),
			config.OptimizerPSInitAdjustTargetWorkerCount: "32",
			config.OptimizerWorkerMemoryMarginPercent:     "0.2",
			config.OptimizerPSMarginCPU:                   "4",
		},
	}

	/////////////////////////////////////////
	plan, err := OptimizeJobPSInitAdjustResource(dataStore, conf, optJob, nil)
	assert.NoError(t, err)
	assert.NotEqual(t, plan, nil)

	optCPU := math.Ceil(0.08*float64(recvOpCount)/2 + 4)

	res := plan.JobRes.TaskGroupResources[common.PSTaskGroupName]
	assert.Equal(t, res.Resource.CPUCore, float32(optCPU))
	assert.Equal(t, res.Count, int32(5))
	assert.Equal(t, res.Resource.Memory, float64(12000))
}

func TestOptimizeJobPSInitialResourceWithHotPS(t *testing.T) {
	dataStore := &dsimpl.BaseDataStore{
		Client: mysql.NewFakeClient(),
	}

	jobUUID := "job-uuid-1"
	cluster := "cluster"

	speed1 := float64(8)
	speed2 := float64(10)

	psCPU1 := 2
	psCPU2 := 24
	stepCount := 5
	psCPUOverload := 0.8

	rtInfo1 := &common.JobRuntimeInfo{
		Speed: speed1,
		PSCPU: map[uint64]float64{
			0: float64(psCPU1),
			1: float64(psCPU2),
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
			0: float64(psCPU1),
			1: float64(psCPU2),
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

	recvOpCount := 249

	modelFeature := &common.ModelFeature{
		RecvOpCount: uint64(recvOpCount),
	}
	modelFeatureStr, err := json.Marshal(modelFeature)
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
			ModelFeature:       string(modelFeatureStr),
		},
	}

	conf := &optconfig.OptimizeAlgorithmConfig{
		CustomizedConfig: map[string]string{
			config.OptimizerStepCountThreshold:            strconv.Itoa(stepCount),
			config.OptimizerPSCPUOverload:                 fmt.Sprintf("%f", psCPUOverload),
			config.OptimizerPSInitAdjustTargetWorkerCount: "32",
			config.OptimizerWorkerMemoryMarginPercent:     "0.2",
			config.OptimizerPSMarginCPU:                   "4",
		},
	}

	/////////////////////////////////////////
	plan, err := OptimizeJobPSInitAdjustResource(dataStore, conf, optJob, nil)
	assert.NoError(t, err)
	assert.NotEqual(t, plan, nil)

	optCPU := psCPU2 + 4

	res := plan.JobRes.TaskGroupResources[common.PSTaskGroupName]
	assert.Equal(t, res.Resource.CPUCore, float32(optCPU))
	assert.Equal(t, res.Count, int32(2))
}

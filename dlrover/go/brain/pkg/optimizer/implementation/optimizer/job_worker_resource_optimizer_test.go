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
	"github.com/intelligent-machine-learning/easydl/brain/pkg/optimizer/implementation/optalgorithm"
	"github.com/stretchr/testify/assert"
	"math"
	"testing"
)

func TestJobWorkerResourceOptimizer(t *testing.T) {
	dataStore := &dsimpl.BaseDataStore{
		Client: mysql.NewFakeClient(),
	}

	jobUUID := "job-uuid-1"
	cluster := "cluster"

	speed1 := float64(8)
	speed2 := float64(10)

	psCPU1 := 1
	psCPU2 := 6
	totalPSCPU := 10

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
	for i := 0; i < defaultStepCountThreshold; i++ {
		rtInfos = append(rtInfos, rtInfo1)
	}
	for i := 0; i < defaultStepCountThreshold; i++ {
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

	jobMetrics := &mysql.JobMetrics{
		UID:                jobUUID,
		JobRuntime:         string(rtInfosStr),
		DatasetFeature:     string(setFeatureStr),
		HyperParamsFeature: string(hyperParamStr),
	}
	dataStore.Client.JobMetricsRecorder.Upsert(jobMetrics)

	jobMetas := []*common.JobMeta{
		{
			UUID:    jobUUID,
			Cluster: cluster,
		},
	}

	///////////////////////////////////////
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

	/////////////////////////////////////////
	optimizerConf := &optconfig.OptimizerConfig{
		OptimizeAlgorithmConfig: &optconfig.OptimizeAlgorithmConfig{
			Name: optalgorithm.OptimizeAlgorithmJobWorkerResource,
			CustomizedConfig: map[string]string{
				config.OptimizerWorkerOptimizePhase: config.OptimizerWorkerOptimizePhaseStable,
			},
		},
	}

	conf := config.NewEmptyConfig()
	conf.Set(config.JobNodeMemoryMarginPercent, 0.4)
	testOptimizer := newJobWorkerResourceOptimizer(dataStore, conf)
	plans, err := testOptimizer.Optimize(optimizerConf, jobMetas)
	assert.NoError(t, err)
	assert.Equal(t, len(plans), 1)
	res := plans[0].AlgOptPlan.JobRes.TaskGroupResources[common.WorkerTaskGroupName]

	replica := defaultPSCPUOverload / (float64(psCPU2) / float64(totalPSCPU)) * float64(len(rtInfo2.WorkerCPU))
	assert.Equal(t, res.Count, int32(math.Ceil(replica)))

	memory := maxMem * 1.2
	cpu := math.Ceil(maxCPU + 1.0)
	assert.Equal(t, res.Resource.Memory, memory)
	assert.Equal(t, res.Resource.CPUCore, float32(cpu))
}

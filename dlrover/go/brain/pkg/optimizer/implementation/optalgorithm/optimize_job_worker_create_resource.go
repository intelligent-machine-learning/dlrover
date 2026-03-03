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
	log "github.com/golang/glog"
	"github.com/intelligent-machine-learning/easydl/brain/pkg/common"
	"github.com/intelligent-machine-learning/easydl/brain/pkg/config"
	datastoreapi "github.com/intelligent-machine-learning/easydl/brain/pkg/datastore/api"
	optconfig "github.com/intelligent-machine-learning/easydl/brain/pkg/optimizer/config"
	optimplcomm "github.com/intelligent-machine-learning/easydl/brain/pkg/optimizer/implementation/common"
	optimplutils "github.com/intelligent-machine-learning/easydl/brain/pkg/optimizer/implementation/utils"
	"math"
	"strconv"
)

const (
	// OptimizeAlgorithmJobWorkerCreateResource is the name of optimize algorithm to optimize the first worker of a job
	OptimizeAlgorithmJobWorkerCreateResource = "optimize_job_worker_create_resource"
	// DefaultWorkerCreateMemory is the default value of memory for the first worker.
	DefaultWorkerCreateMemory = 16 * 1024 * 1024 * 1024 // 16G
)

func init() {
	registerOptimizeAlgorithm(OptimizeAlgorithmJobWorkerCreateResource, OptimizeJobWorkerCreateResource)
}

// OptimizeJobWorkerCreateResource optimizes job first worker resources
func OptimizeJobWorkerCreateResource(dataStore datastoreapi.DataStore, conf *optconfig.OptimizeAlgorithmConfig, optJob *common.OptimizeJobMeta,
	historyJobs []*common.OptimizeJobMeta) (*common.AlgorithmOptimizePlan, error) {

	maxMemory := 0.0
	maxCPUCore := 0.0

	for _, historyJob := range historyJobs {
		job := historyJob.Metrics
		if job.ExitReason != optimplcomm.ExitReasonCompleted {
			continue
		}
		maxJobCPUCore := 0.0
		maxJobMemory := 0.0

		runtimes := make([]*common.JobRuntimeInfo, 0)
		err := json.Unmarshal([]byte(job.JobRuntime), &runtimes)
		if err != nil {
			log.Errorf("fail to unmarshal job %s runtime: %v", job.JobName, err)
			continue
		}

		for _, runtime := range runtimes {
			memory := optimplutils.GetMaxJobNodeResource(runtime.WorkerMemory)
			cpuCore := optimplutils.GetMaxJobNodeResource(runtime.WorkerCPU)
			if maxJobMemory < memory {
				maxJobMemory = memory
			}
			if maxJobCPUCore < cpuCore {
				maxJobCPUCore = cpuCore
			}
		}

		if maxMemory < maxJobMemory {
			maxMemory = maxJobMemory
		}
		if maxCPUCore < maxJobCPUCore {
			maxCPUCore = maxJobCPUCore
		}
	}

	memoryMarginPercent, err := strconv.ParseFloat(conf.CustomizedConfig[config.JobNodeMemoryMarginPercent], 64)
	if err != nil || memoryMarginPercent == 0 {
		memoryMarginPercent = optimplcomm.DefaultMemoryMarginPercent
	}
	minChiefCPUCores, err := strconv.ParseFloat(conf.CustomizedConfig[config.OptimizerMinWorkerCreateCPU], 64)
	if err != nil || minChiefCPUCores == 0 {
		minChiefCPUCores = optimplcomm.DefaultWorkerCreateCPU
	}
	memory := maxMemory * (1 + memoryMarginPercent)
	cpuCore := math.Ceil(maxCPUCore)

	if memory < DefaultWorkerCreateMemory {
		memory = DefaultWorkerCreateMemory
	}

	// Set the enough CPU of chief to measure the cpu requirement.
	if cpuCore < minChiefCPUCores {
		cpuCore = minChiefCPUCores
	}

	resOptPlan := &common.AlgorithmOptimizePlan{
		JobRes: &common.JobResource{
			TaskGroupResources: map[string]*common.TaskGroupResource{
				common.WorkerTaskGroupName: {
					Count: 1,
					Resource: &common.PodResource{
						CPUCore: float32(cpuCore),
						Memory:  memory,
					},
				},
			},
		},
	}

	return resOptPlan, nil
}

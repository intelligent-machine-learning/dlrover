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
	"fmt"
	log "github.com/golang/glog"
	"github.com/intelligent-machine-learning/easydl/brain/pkg/common"
	"github.com/intelligent-machine-learning/easydl/brain/pkg/config"
	datastoreapi "github.com/intelligent-machine-learning/easydl/brain/pkg/datastore/api"
	optconfig "github.com/intelligent-machine-learning/easydl/brain/pkg/optimizer/config"
	"strconv"
)

const (
	// OptimizeAlgorithmJobPSColdCreateResource is the name of cold job ps create resource optimize function
	OptimizeAlgorithmJobPSColdCreateResource = "optimize_job_ps_cold_create_resource"
)

func init() {
	registerOptimizeAlgorithm(OptimizeAlgorithmJobPSColdCreateResource, OptimizeJobPSColdCreateResource)
}

// OptimizeJobPSColdCreateResource optimizes cold job ps initial resources
func OptimizeJobPSColdCreateResource(dataStore datastoreapi.DataStore, conf *optconfig.OptimizeAlgorithmConfig, optJob *common.OptimizeJobMeta,
	historyJobs []*common.OptimizeJobMeta) (*common.AlgorithmOptimizePlan, error) {

	if conf == nil || conf.CustomizedConfig == nil {
		err := fmt.Errorf("Invalid config: %v", conf)
		return nil, err
	}

	coldReplica, err := strconv.Atoi(conf.CustomizedConfig[config.OptimizerPSColdReplica])
	if err != nil {
		log.Errorf("Fail to get cold ps replica %s: %v", conf.CustomizedConfig[config.OptimizerPSColdReplica], err)
		return nil, err
	}
	coldCPU, err := strconv.ParseFloat(conf.CustomizedConfig[config.OptimizerPSColdCPU], 64)
	if err != nil {
		log.Errorf("Fail to get cold ps cpu %s: %v", conf.CustomizedConfig[config.OptimizerPSColdCPU], err)
		return nil, err
	}
	coldMemory, err := strconv.ParseFloat(conf.CustomizedConfig[config.OptimizerPSColdMemory], 64)
	if err != nil {
		log.Errorf("Fail to get cold ps memory %s: %v", conf.CustomizedConfig[config.OptimizerPSColdMemory], err)
		return nil, err
	}

	resOptPlan := &common.AlgorithmOptimizePlan{
		JobRes: &common.JobResource{
			TaskGroupResources: map[string]*common.TaskGroupResource{
				common.PSTaskGroupName: {
					Count: int32(coldReplica),
					Resource: &common.PodResource{
						CPUCore: float32(coldCPU),
						Memory:  coldMemory,
						GPUCore: 0,
						GPUType: "",
					},
				},
			},
		},
	}

	return resOptPlan, nil
}

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
	log "github.com/golang/glog"
	"github.com/intelligent-machine-learning/easydl/brain/pkg/common"
	"github.com/intelligent-machine-learning/easydl/brain/pkg/config"
	datastoreapi "github.com/intelligent-machine-learning/easydl/brain/pkg/datastore/api"
	"github.com/intelligent-machine-learning/easydl/brain/pkg/datastore/recorder/mysql"
	optconfig "github.com/intelligent-machine-learning/easydl/brain/pkg/optimizer/config"
	optimplcomm "github.com/intelligent-machine-learning/easydl/brain/pkg/optimizer/implementation/common"
	"strconv"
)

const (
	// OptimizeAlgorithmJobPSOomResource is the name of job ps oom resource optimize function
	OptimizeAlgorithmJobPSOomResource = "optimize_job_ps_oom_resource"
)

func init() {
	registerOptimizeAlgorithm(OptimizeAlgorithmJobPSOomResource, OptimizeJobPSOomResource)
}

// OptimizeJobPSOomResource optimizes job ps resources when ps is oom
func OptimizeJobPSOomResource(dataStore datastoreapi.DataStore, conf *optconfig.OptimizeAlgorithmConfig, optJob *common.OptimizeJobMeta,
	historyJobs []*common.OptimizeJobMeta) (*common.AlgorithmOptimizePlan, error) {

	if conf == nil || conf.CustomizedConfig == nil {
		err := fmt.Errorf("invalid config: %v", conf)
		return nil, err
	}

	workloadUnbalancePercent, err := strconv.ParseFloat(conf.CustomizedConfig[config.OptimizerPSMemoryWorkloadUnbalancePercent], 64)
	if err != nil {
		log.Errorf("fail to get OptimizerPSMemoryWorkloadBalancePercent %s: %v", conf.CustomizedConfig[config.OptimizerPSMemoryWorkloadUnbalancePercent], err)
		return nil, err
	}

	optMemory := 0.0
	replica := 0
	optCPU := 0.0

	cond := &datastoreapi.Condition{
		Type: common.TypeGetDataListJobNode,
		Extra: &mysql.JobNodeCondition{
			JobUUID: optJob.JobMeta.UUID,
			Type:    common.PSTaskGroupName,
		},
	}
	nodes := make([]*mysql.JobNode, 0)
	err = dataStore.GetData(cond, &nodes)
	if err != nil {
		log.Errorf("fail to get nodes of job %s: %v", optJob.JobMeta.Name, err)
		return nil, err
	}

	currReplica := 0
	for _, node := range nodes {
		status := &common.JobNodeStatus{}
		err = json.Unmarshal([]byte(node.Status), status)
		if err != nil {
			log.Errorf("fail to unmarshal status %s of job node %s: %v", node.Status, node.Name, err)
			continue
		}

		resource := &common.PodResource{}
		err = json.Unmarshal([]byte(node.Resource), resource)
		if err != nil {
			log.Errorf("fail to unmarshal resource %s of job node %s: %v", node.Resource, node.Name, err)
			continue
		}

		if status.Status == "Running" || status.IsOOM {
			currReplica++
		}

		memory := resource.Memory
		if memory > optMemory {
			optMemory = memory
		}
		cpu := float64(resource.CPUCore)
		if cpu > optCPU {
			optCPU = cpu
		}
	}

	jobMetrics := optJob.Metrics
	if len(jobMetrics.JobRuntime) == 0 { // no runtime information collected
		if optMemory >= optimplcomm.DefaultMaxPSMemory {
			replica = currReplica * 2
		} else {
			optMemory = optMemory * 2
		}
	} else {
		rts := make([]*common.JobRuntimeInfo, 0)
		err = json.Unmarshal([]byte(jobMetrics.JobRuntime), &rts)
		if err != nil {
			log.Errorf("fail to unmarshal runtime infos of %s from %s: %v", optJob.JobMeta.Name, jobMetrics.JobRuntime, err)
			return nil, err
		}
		l := len(rts)
		if l == 0 {
			log.Warningf("no runtime infos of %s from %s", optJob.JobMeta.Name, jobMetrics.JobRuntime)
			return nil, nil
		}

		lastRt := rts[l-1]
		totalMemory := 0.0
		maxMemory := 0.0
		for _, memory := range lastRt.PSMemory {
			if maxMemory < memory {
				maxMemory = memory
			}
			totalMemory = totalMemory + memory
		}
		currReplica = len(lastRt.PSMemory)
		avgMemory := totalMemory / float64(currReplica)
		if (maxMemory-avgMemory)/maxMemory > workloadUnbalancePercent {
			optMemory = maxMemory * 2
		} else {
			replica = currReplica * 2
		}
	}

	resOptPlan := &common.AlgorithmOptimizePlan{
		JobRes: &common.JobResource{
			TaskGroupResources: map[string]*common.TaskGroupResource{
				common.PSTaskGroupName: {
					Count: int32(replica),
					Resource: &common.PodResource{
						Memory:  optMemory,
						CPUCore: float32(optCPU),
					},
				},
			},
		},
	}

	return resOptPlan, nil
}

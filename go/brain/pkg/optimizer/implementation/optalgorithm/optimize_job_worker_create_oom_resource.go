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
	"github.com/intelligent-machine-learning/easydl/brain/pkg/datastore/recorder/mysql"
	optconfig "github.com/intelligent-machine-learning/easydl/brain/pkg/optimizer/config"
	optimplutils "github.com/intelligent-machine-learning/easydl/brain/pkg/optimizer/implementation/utils"
	pb "github.com/intelligent-machine-learning/easydl/brain/pkg/proto"
	"github.com/intelligent-machine-learning/easydl/brain/pkg/utils"
	"strconv"
	"strings"
)

const (
	// OptimizeAlgorithmJobWorkerCreateOomResource is the name of optimize algorithm to optimize the first worker which is oom before
	OptimizeAlgorithmJobWorkerCreateOomResource = "optimize_job_worker_create_oom_resource"
)

func init() {
	registerOptimizeAlgorithm(OptimizeAlgorithmJobWorkerCreateOomResource, OptimizeJobWorkerCreateOomResource)
}

// OptimizeJobWorkerCreateOomResource optimizes the first worker which is oom before
func OptimizeJobWorkerCreateOomResource(dataStore datastoreapi.DataStore, conf *optconfig.OptimizeAlgorithmConfig, optJob *common.OptimizeJobMeta,
	historyJobs []*common.OptimizeJobMeta) (*common.AlgorithmOptimizePlan, error) {

	oomMemoryMarginPercent, err := strconv.ParseFloat(conf.CustomizedConfig[config.OptimizerWorkerOomMemoryMarginPercent], 64)
	if err != nil {
		log.Errorf("fail to get OptimizerWorkerOomMemoryMarginPercent %s: %v", conf.CustomizedConfig[config.OptimizerWorkerOomMemoryMarginPercent], err)
		return nil, err
	}
	minMemoryIncrease, err := strconv.ParseFloat(conf.CustomizedConfig[config.OptimizerWorkerOomMemoryMinIncrease], 64)
	if err != nil {
		log.Errorf("fail to get OptimizerWorkerOomMemoryMinIncrease %s: %v", conf.CustomizedConfig[config.OptimizerWorkerOomMemoryMinIncrease], err)
		return nil, err
	}

	maxMemory := 0.0

	for _, historyJob := range historyJobs {
		job := historyJob.Metrics
		rts := make([]*common.JobRuntimeInfo, 0)
		err = json.Unmarshal([]byte(job.JobRuntime), &rts)
		if err != nil {
			log.Errorf("fail to unmarshal job runtime for %s: %v", job.JobName, err)
			continue
		}

		cond := &datastoreapi.Condition{
			Type: common.TypeGetDataListJobNode,
			Extra: &mysql.JobNodeCondition{
				JobUUID: job.JobUUID,
				Type:    common.WorkerTaskGroupName,
			},
		}
		nodes := make([]*mysql.JobNode, 0)
		err = dataStore.GetData(cond, &nodes)
		if err != nil {
			return nil, err
		}

		l := len(rts)
		for _, node := range nodes {
			memory := 0.0
			_, nodeID := utils.ExtractPodTypeAndIDFromName(node.Name)
			if nodeID < 0 {
				log.Errorf("fail to extract id for node %s", node.Name)
				continue
			}
			for i := l - 1; i >= 0; i-- {
				mem, found := rts[i].WorkerMemory[uint64(nodeID)]
				if found {
					memory = mem
					break
				}
			}
			if memory == 0 {
				log.Errorf("fail to have the memory for node %s", node.Name)
				continue
			}

			nodeStatus := &common.JobNodeStatus{}
			err = json.Unmarshal([]byte(node.Status), nodeStatus)
			if err != nil {
				log.Errorf("fail to unmarshal node %s status %s: %v", node.Name, node.Status, err)
				continue
			}
			if nodeStatus.IsOOM {
				memory = memory * (1 + oomMemoryMarginPercent)
			}
			if memory > maxMemory {
				maxMemory = memory
			}
		}
	}

	lastOptWorkerMemory := 0.0
	optResults := make([]*pb.JobOptimization, 0)
	err = json.Unmarshal([]byte(optJob.Metrics.Optimization), &optResults)
	if err == nil {
		l := len(optResults)
		for i := l - 1; i >= 0; i-- {
			lastOptWorkerMemory = getWorkerMemoryFromOptimizeResult(optResults[i])
			if lastOptWorkerMemory > 0 {
				break
			}
		}
	} else {
		log.Warningf("fail to unmarshal job %s optimization info %s: %v", optJob.JobMeta.Name, optJob.Metrics.Optimization, err)
	}

	if lastOptWorkerMemory == 0 {
		rts := make([]*common.JobRuntimeInfo, 0)
		err = json.Unmarshal([]byte(optJob.Metrics.JobRuntime), &rts)
		if err == nil && len(rts) > 0 {
			l := len(rts)
			rt := rts[l-1]
			lastOptWorkerMemory = optimplutils.GetMaxJobNodeResource(rt.WorkerMemory)
		} else {
			log.Warningf("fail to unmarshal job %s runtime info %s: %v", optJob.JobMeta.Name, optJob.Metrics.JobRuntime, err)
		}
	}

	if lastOptWorkerMemory > 0 && lastOptWorkerMemory+minMemoryIncrease > maxMemory {
		maxMemory = lastOptWorkerMemory + minMemoryIncrease
	}
	resOptPlan := &common.AlgorithmOptimizePlan{
		JobRes: &common.JobResource{
			TaskGroupResources: map[string]*common.TaskGroupResource{
				common.WorkerTaskGroupName: {
					Resource: &common.PodResource{
						Memory: maxMemory,
					},
				},
			},
		},
	}

	return resOptPlan, nil
}

func getWorkerMemoryFromOptimizeResult(result *pb.JobOptimization) float64 {
	if result == nil || result.Plan == nil || result.Plan.Resource == nil {
		return 0.0
	}
	maxMemory := 0.0
	res := result.Plan.Resource
	if res.PodResources != nil {
		for name, podRes := range res.PodResources {
			if strings.Contains(name, "worker-") {
				memory := float64(podRes.Memory)
				if maxMemory < memory {
					maxMemory = memory
				}
			}
		}
	}

	if res.TaskGroupResources != nil {
		taskGroupRes, found := res.TaskGroupResources[common.WorkerTaskGroupName]
		if found && taskGroupRes.Resource != nil {
			memory := float64(taskGroupRes.Resource.Memory)
			if maxMemory < memory {
				maxMemory = memory
			}
		}
	}
	return maxMemory
}

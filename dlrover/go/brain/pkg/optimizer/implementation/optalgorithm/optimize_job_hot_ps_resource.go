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
	optimplcomm "github.com/intelligent-machine-learning/easydl/brain/pkg/optimizer/implementation/common"
	optimplutils "github.com/intelligent-machine-learning/easydl/brain/pkg/optimizer/implementation/utils"
	"github.com/intelligent-machine-learning/easydl/brain/pkg/utils"
	"math"
	"strconv"
)

const (
	// OptimizeAlgorithmJobHotPSResource is the name of running job ps resource optimize function
	OptimizeAlgorithmJobHotPSResource = "optimize_job_hot_ps_resource"
	psPodNameSuffix                   = "-edljob-ps-"
	maxCPUThreshold                   = 32
)

func init() {
	registerOptimizeAlgorithm(OptimizeAlgorithmJobHotPSResource, OptimizeJobHotPSResource)
}

// OptimizeJobHotPSResource optimizes job ps initial resources
func OptimizeJobHotPSResource(dataStore datastoreapi.DataStore, conf *optconfig.OptimizeAlgorithmConfig, optJob *common.OptimizeJobMeta,
	historyJobs []*common.OptimizeJobMeta) (*common.AlgorithmOptimizePlan, error) {

	hotCPUThreshold, err := strconv.ParseFloat(conf.CustomizedConfig[config.OptimizerHotPSCPUThreshold], 64)
	if err != nil {
		log.Errorf("fail to get ps cpu hot threshold %s: %v", conf.CustomizedConfig[config.OptimizerHotPSCPUThreshold], err)
		return nil, err
	}

	hotMemoryThreshold, err := strconv.ParseFloat(conf.CustomizedConfig[config.OptimizerHotPSMemoryThreshold], 64)
	if err != nil {
		log.Errorf("fail to get ps cpu hot threshold %s: %v", conf.CustomizedConfig[config.OptimizerHotPSMemoryThreshold], err)
		return nil, err
	}

	hotTargetWorkerCount, err := strconv.Atoi(conf.CustomizedConfig[config.OptimizerHotPSCPUTargetWorkerCount])
	if err != nil {
		log.Errorf("fail to get the target worker count %s: %v", conf.CustomizedConfig[config.OptimizerHotPSCPUTargetWorkerCount], err)
		return nil, err
	}

	hotAdjustMemory, err := strconv.Atoi(conf.CustomizedConfig[config.OptimizerHotPSMemoryAdjust])
	if err != nil {
		log.Errorf("fail to get ps cpu hot adjustment %s: %v", conf.CustomizedConfig[config.OptimizerHotPSMemoryAdjust], err)
		return nil, err
	}

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
		log.Errorf("fail to get nodes for %s: %v", optJob.JobMeta.UUID, err)
		return nil, err
	}

	nodeCPUs := make(map[uint64]float64)
	nodeMemory := make(map[uint64]float64)
	nodeNames := make(map[uint64]string)
	for _, node := range nodes {
		_, num := utils.ExtractPodTypeAndIDFromName(node.Name)
		if num < 0 {
			log.Errorf("fail to extract number for %s", node.Name)
			continue
		}
		res := &common.PodResource{}
		err = json.Unmarshal([]byte(node.Resource), res)
		if err != nil {
			log.Errorf("fail to unmarshal resource %s for node %s: %v", node.Resource, node.Name, err)
			continue
		}

		num64 := uint64(num)
		nodeCPUs[num64] = float64(res.CPUCore)
		nodeMemory[num64] = res.Memory
		nodeNames[num64] = node.Name
	}

	runtimeInfos := make([]*common.JobRuntimeInfo, 0)
	err = json.Unmarshal([]byte(optJob.Metrics.JobRuntime), &runtimeInfos)
	if err != nil {
		log.Errorf("fail to unmarshal runtime info for %s: %v", optJob.JobMeta.Name, err)
		return nil, err
	}

	if len(runtimeInfos) == 0 {
		log.Info("there is no runtime infos")
		return nil, nil
	}

	runtimeInfos = optimplutils.FilterRuntimeInfosWithLatestPS(runtimeInfos)

	optNodeRes := make(map[string]*common.PodResource)

	hotCPUPsNodes := optimplutils.CheckHotCPUNodes(runtimeInfos, nodeCPUs, hotCPUThreshold, optimplcomm.NRecordToAvgResource)
	hotMemoryPsNodes := checkHotMemoryNodes(runtimeInfos, nodeMemory, hotMemoryThreshold, optimplcomm.NRecordToAvgResource)

	if len(hotCPUPsNodes) > 0 {
		rt := runtimeInfos[len(runtimeInfos)-1]
		curWorkerNum := len(rt.WorkerCPU)
		avgCPU := optimplutils.CalculateJobNodeAvgResources(runtimeInfos, optimplcomm.NRecordToAvgResource, optimplcomm.ResourceTypePSCPU)

		coeff := float64(hotTargetWorkerCount) / float64(curWorkerNum)
		for _, n := range hotCPUPsNodes {
			optCPU := math.Ceil(avgCPU[n] * coeff)
			if optCPU > maxCPUThreshold {
				optCPU = maxCPUThreshold
				coeff = optCPU / avgCPU[n]
			}
		}

		// Enlarge the CPU of ps nodes with the same ratio.
		for n, cpu := range avgCPU {
			optCPU := math.Ceil(cpu * coeff)
			if optCPU > nodeCPUs[n] {
				nodeName := nodeNames[n]
				optNodeRes[nodeName] = &common.PodResource{
					CPUCore: float32(optCPU),
				}
			}
		}
	}

	for _, n := range hotMemoryPsNodes {
		totalMemory, found := nodeMemory[n]
		if !found {
			log.Errorf("fail to find task %d total cpu", n)
			continue
		}
		nodeName := nodeNames[n]
		optMemory := totalMemory + float64(hotAdjustMemory)
		if _, ok := optNodeRes[nodeName]; ok {
			optNodeRes[nodeName].Memory = optMemory
		} else {
			optNodeRes[nodeName] = &common.PodResource{
				Memory: optMemory,
			}
		}
	}

	if len(optNodeRes) == 0 {
		return nil, nil
	}

	return &common.AlgorithmOptimizePlan{
		JobRes: &common.JobResource{
			PodResources: optNodeRes,
		},
	}, nil
}

func checkHotMemoryNodes(runtimeInfos []*common.JobRuntimeInfo, nodeMemory map[uint64]float64, hotThreshold float64, checkHotStep int) []uint64 {
	if len(runtimeInfos) < checkHotStep {
		return nil
	}
	hotPsRecords := make(map[uint64]int32)
	rt := runtimeInfos[len(runtimeInfos)-1]
	for n := range rt.PSMemory {
		hotPsRecords[n] = 0
	}

	for i := 0; i < checkHotStep; i++ {
		rt := runtimeInfos[len(runtimeInfos)-i-1]
		for n, memory := range rt.PSMemory {
			totalMemory, found := nodeMemory[n]
			if !found {
				log.Errorf("fail to find task %d total memory", n)
				continue
			}
			memUtil := memory / totalMemory
			if memUtil > hotThreshold {
				hotPsRecords[n]++
				log.Infof("The memory util %f of PS %d is overload %f ", memUtil, n, hotThreshold)
			}
		}
	}
	hotPsIds := make([]uint64, 0)
	for n, num := range hotPsRecords {
		if num >= int32(checkHotStep) {
			hotPsIds = append(hotPsIds, n)
		}
	}
	return hotPsIds
}

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
	// OptimizeAlgorithmJobPSResourceUtil is the name of running job ps resource optimize function
	OptimizeAlgorithmJobPSResourceUtil = "optimize_job_ps_resource_util"
	remainingTimeThreshold             = 3600 // 1h
)

func init() {
	registerOptimizeAlgorithm(OptimizeAlgorithmJobPSResourceUtil, OptimizeJobPSResourceUtil)
}

// OptimizeJobPSResourceUtil optimizes job ps resurce utilization
func OptimizeJobPSResourceUtil(dataStore datastoreapi.DataStore, conf *optconfig.OptimizeAlgorithmConfig, optJob *common.OptimizeJobMeta,
	historyJobs []*common.OptimizeJobMeta) (*common.AlgorithmOptimizePlan, error) {

	lowCPUThreshold, err := strconv.ParseFloat(conf.CustomizedConfig[config.OptimizerLowPSCPUThreshold], 64)
	if err != nil {
		log.Errorf("fail to get ps cpu low threshold %s: %v", conf.CustomizedConfig[config.OptimizerLowPSCPUThreshold], err)
		return nil, err
	}

	memoryMarginPercent, err := strconv.ParseFloat(conf.CustomizedConfig[config.OptimizerPSMemoryMarginPercent], 64)
	if err != nil {
		log.Errorf("fail to get ps memory margin percent %s: %v", conf.CustomizedConfig[config.OptimizerPSMemoryMarginPercent], err)
		return nil, err
	}

	marginCPU, err := strconv.Atoi(conf.CustomizedConfig[config.OptimizerPSMarginCPU])
	if err != nil {
		log.Errorf("fail to get ps margin cpu %s: %v", conf.CustomizedConfig[config.OptimizerPSMarginCPU], err)
		return nil, err
	}

	psCPUOverload, err := strconv.ParseFloat(conf.CustomizedConfig[config.OptimizerPSCPUOverload], 64)
	if err != nil {
		log.Errorf("Fail to get ps cpu overload %s: %v", conf.CustomizedConfig[config.OptimizerPSCPUOverload], err)
		return nil, err
	}

	workerCountThreshold, err := strconv.Atoi(conf.CustomizedConfig[config.OptimizerHotPSCPUTargetWorkerCount])
	if err != nil {
		log.Errorf("Fail to get worker max init count per step %s: %v", conf.CustomizedConfig[config.OptimizerHotPSCPUTargetWorkerCount], err)
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
		log.Errorf("fail to get tasks for %s: %v", optJob.JobMeta.UUID, err)
		return nil, err
	}

	nodeCPU := make(map[uint64]float64)
	nodeMemory := make(map[uint64]float64)
	nodeName := make(map[uint64]string)
	for _, node := range nodes {
		_, num := utils.ExtractPodTypeAndIDFromName(node.Name)
		if num < 0 {
			log.Errorf("fail to extract number for %s", node.Name)
			continue
		}
		num64 := uint64(num)

		res := &common.PodResource{}
		err = json.Unmarshal([]byte(node.Resource), res)
		if err != nil {
			log.Errorf("fail to unmarshal resource %s for %s: %v", node.Resource, node.Name, err)
			continue
		}

		nodeCPU[num64] = float64(res.CPUCore)
		nodeMemory[num64] = res.Memory
		nodeName[num64] = node.Name
	}

	runtimeInfos := make([]*common.JobRuntimeInfo, 0)
	err = json.Unmarshal([]byte(optJob.Metrics.JobRuntime), &runtimeInfos)
	if err != nil {
		log.Errorf("fail to unmarshal runtime info for %s: %v", optJob.JobMeta.Name, err)
		return nil, err
	}
	runtimeInfos = optimplutils.FilterRuntimeInfosWithLatestPS(runtimeInfos)

	if len(runtimeInfos) < optimplcomm.NRecordToAvgResource {
		log.Info("there is no enough runtime infos")
		return nil, nil
	}
	latestRuntimeInfo := runtimeInfos[len(runtimeInfos)-1]

	remainingTime := estimateRemainingRunningTime(optJob.Metrics, runtimeInfos, optimplcomm.NRecordToAvgResource)
	if remainingTime < remainingTimeThreshold {
		log.Infof("The remaining time is %f less than %d", remainingTime, remainingTimeThreshold)
		return nil, nil
	}
	psAvgCPU := optimplutils.CalculateJobNodeAvgResources(runtimeInfos, optimplcomm.NRecordToAvgResource, optimplcomm.ResourceTypePSCPU)
	maxPSCPUUtil := optimplutils.GetMaxUtil(psAvgCPU, nodeCPU)
	currentWorkerCount := len(latestRuntimeInfo.WorkerCPU)

	enabledOptimizeResourceUtil := false
	if currentWorkerCount >= workerCountThreshold && maxPSCPUUtil > psCPUOverload {
		// DLRover will not increase the number of workers when we can reduce CPU cores
		// of some PS nodes whose CPU utilization is low.
		enabledOptimizeResourceUtil = true
		log.Infof("Enable optimizing PS CPU util due to maxPSCPUUtil %f > psCPUOverload %f", maxPSCPUUtil, psCPUOverload)
	}
	maxOverloadCPU := maxCPUThreshold * psCPUOverload
	for _, cpu := range psAvgCPU {
		if cpu >= maxOverloadCPU {
			// CPU cores of a PS has reached the maximum and we can reduce CPU cores
			// of some PS nodes whose CPU utilization is low.
			enabledOptimizeResourceUtil = true
			log.Infof("Enable optimizing PS CPU util due to cpu %f > CPUOverload %f", cpu, maxOverloadCPU)
		}
	}
	if !enabledOptimizeResourceUtil {
		return nil, nil
	}

	optTaskRes := make(map[string]*common.PodResource)

	psMaxCPU := optimplutils.CalculateJobNodeMaxResource(runtimeInfos, optimplcomm.NRecordToAvgResource, optimplcomm.ResourceTypePSCPU)
	lowCPUPsNodes := checkLowCPUUtilNodes(psMaxCPU, nodeCPU, lowCPUThreshold, optimplcomm.NRecordToAvgResource)

	for _, ps := range lowCPUPsNodes {
		name := nodeName[ps]
		optCPU := math.Ceil(psMaxCPU[ps]) + float64(marginCPU)
		optMemory := latestRuntimeInfo.PSMemory[ps] * (1.0 + memoryMarginPercent)

		optTaskRes[name] = &common.PodResource{
			CPUCore: float32(optCPU),
			Memory:  optMemory,
		}
	}

	if len(optTaskRes) == 0 {
		return nil, nil
	}

	return &common.AlgorithmOptimizePlan{
		JobRes: &common.JobResource{
			PodResources: optTaskRes,
		},
	}, nil
}

func checkLowCPUUtilNodes(psUsedCPU map[uint64]float64, taskCPU map[uint64]float64, lowThreshold float64, sampleCount int) []uint64 {
	lowPsIds := make([]uint64, 0)

	for ps, cpu := range psUsedCPU {
		requestedCPU, found := taskCPU[ps]
		if !found {
			log.Errorf("Fail to find the requested CPU of task %d", ps)
			continue
		}
		cpuUtil := cpu / requestedCPU
		if cpuUtil < lowThreshold {
			lowPsIds = append(lowPsIds, ps)
		}
	}
	return lowPsIds
}

func estimateRemainingRunningTime(metrics *common.JobMetrics, runtimeInfos []*common.JobRuntimeInfo, maxCount int32) float64 {
	l := int32(len(runtimeInfos))
	latestRuntimeInfo := runtimeInfos[l-1]

	trainingHyperParam := &common.TrainingHyperParams{}
	err := json.Unmarshal([]byte(metrics.HyperParamsFeature), trainingHyperParam)
	if err != nil {
		log.Errorf("Fail to unmarshal hyper parameters: %v", err)
		return 0
	}
	batchSize := float64(trainingHyperParam.BatchSize)

	trainingDatasetFeature := &common.TrainingSetFeature{}
	err = json.Unmarshal([]byte(metrics.DatasetFeature), trainingDatasetFeature)
	if err != nil {
		log.Errorf("Fail to unmarshal dataset feature: %v", err)
		return 0
	}
	datasetSize := float64(trainingDatasetFeature.DatasetSize)

	steps := datasetSize / batchSize
	epoch := float64(trainingHyperParam.Epoch)
	maxSteps := float64(trainingHyperParam.MaxSteps)
	if epoch > 0 {
		steps = epoch * steps
	}
	if maxSteps > 0 {
		if steps > maxSteps {
			steps = maxSteps
		}
	}
	remainingStep := steps - float64(latestRuntimeInfo.GlobalStep)

	totalSpeed := 0.0
	for i := l - 1; i >= l-maxCount; i-- {
		totalSpeed = totalSpeed + runtimeInfos[i].Speed
	}
	avgSpeed := totalSpeed / float64(maxCount)

	time := remainingStep / avgSpeed
	return time
}

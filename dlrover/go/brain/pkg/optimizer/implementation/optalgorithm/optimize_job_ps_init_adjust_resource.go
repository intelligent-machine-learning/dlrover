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
	optconfig "github.com/intelligent-machine-learning/easydl/brain/pkg/optimizer/config"
	optimplcomm "github.com/intelligent-machine-learning/easydl/brain/pkg/optimizer/implementation/common"
	optimplutils "github.com/intelligent-machine-learning/easydl/brain/pkg/optimizer/implementation/utils"
	"math"
	"strconv"
)

const (
	// OptimizeAlgorithmJobPSInitAdjustResource is the name of algorithm which adjust job ps resource when job is just running
	OptimizeAlgorithmJobPSInitAdjustResource = "optimize_job_ps_init_adjust_resource"
)

func init() {
	registerOptimizeAlgorithm(OptimizeAlgorithmJobPSInitAdjustResource, OptimizeJobPSInitAdjustResource)
}

// OptimizeJobPSInitAdjustResource optimizes job ps resource when the job is just running
func OptimizeJobPSInitAdjustResource(dataStore datastoreapi.DataStore, conf *optconfig.OptimizeAlgorithmConfig, optJob *common.OptimizeJobMeta,
	historyJobs []*common.OptimizeJobMeta) (*common.AlgorithmOptimizePlan, error) {

	stepCountThreshold, err := strconv.Atoi(conf.CustomizedConfig[config.OptimizerStepCountThreshold])
	if err != nil {
		log.Errorf("Fail to get step count threshold %s: %v", conf.CustomizedConfig[config.OptimizerStepCountThreshold], err)
		return nil, err
	}

	targetWorkerCount, err := strconv.Atoi(conf.CustomizedConfig[config.OptimizerPSInitAdjustTargetWorkerCount])
	if err != nil {
		log.Errorf("Fail to get worker max init count per step %s: %v", conf.CustomizedConfig[config.OptimizerPSInitAdjustTargetWorkerCount], err)
		return nil, err
	}

	marginCPU, err := strconv.Atoi(conf.CustomizedConfig[config.OptimizerPSMarginCPU])
	if err != nil {
		log.Errorf("fail to get ps margin cpu %s: %v", conf.CustomizedConfig[config.OptimizerPSMarginCPU], err)
		return nil, err
	}

	memoryMarginPercent, err := strconv.ParseFloat(conf.CustomizedConfig[config.OptimizerPSMemoryMarginPercent], 64)
	if err != nil || memoryMarginPercent == 0 {
		memoryMarginPercent = optimplcomm.DefaultPSMemoryMarginPercent
	}

	jobMetrics := optJob.Metrics

	modelFeature := &common.ModelFeature{}
	err = json.Unmarshal([]byte(jobMetrics.ModelFeature), modelFeature)
	if err != nil {
		log.Errorf("fail to unmarshal model feature of %s: %v", optJob.JobMeta.Name, err)
		return nil, err
	}

	rts := make([]*common.JobRuntimeInfo, 0)
	err = json.Unmarshal([]byte(jobMetrics.JobRuntime), &rts)
	if err != nil {
		log.Errorf("fail to unmarshal runtime infos of %s: %v", optJob.JobMeta.Name, err)
		return nil, err
	}

	l := len(rts)
	if l == 0 {
		err = fmt.Errorf("job %s does not have enough runtime infos", optJob.JobMeta.Name)
		return nil, err
	}
	latestPSCPU := rts[l-1].PSCPU
	currPSCount := float64(len(latestPSCPU))
	psAvgCPU := optimplutils.CalculateJobNodeAvgResources(rts, optimplcomm.NRecordToAvgResource, optimplcomm.ResourceTypePSCPU)

	avgSpeed := optimplutils.ComputeAvgSpeed(rts, stepCountThreshold)
	workerTargetReplica := 0.0
	if avgSpeed > float64(0) {
		time, err := optimplutils.ComputePerStepTime(jobMetrics, avgSpeed)
		if err != nil {
			log.Errorf("fail to compute job %s initial step time: %v", optJob.JobMeta.Name, err)
		} else if time <= initStepTime {
			workerTargetReplica = float64(defaultInitWorker)
		} else {
			workerTargetReplica = float64(targetWorkerCount)
		}
	}

	recvOpPerPS := float64(modelFeature.RecvOpCount) / currPSCount

	psCPU := float64(16)
	if recvOpPerPS <= 150 {
		psCPU = math.Ceil(0.08*recvOpPerPS) + float64(marginCPU)
	}

	maxPSCPU := math.Ceil(optimplutils.GetMaxJobNodeResource(psAvgCPU))
	if psCPU < maxPSCPU+float64(marginCPU) {
		psCPU = maxPSCPU + float64(marginCPU)
	}

	maxSumUsedCPU := 0.0

	for _, rt := range rts {
		sumUsedCPU := getSumUsedCores(rt.PSCPU)
		if sumUsedCPU > maxSumUsedCPU {
			maxSumUsedCPU = sumUsedCPU
		}
	}

	maxUsedMemory := 0.0
	for _, memory := range rts[len(rts)-1].PSMemory {
		if memory > maxUsedMemory {
			maxUsedMemory = memory
		}
	}

	workerCount := float64(len(rts[l-1].WorkerCPU))

	// After scaling up the number of PS, the workload of the PS will be shared by
	// other PS servers. So, the max CPU cores of the PS will be shrinken.
	estimateMaxPSCPU := maxPSCPU / (float64(optimplcomm.DefaultMaxPSCount) / currPSCount)
	estimatePSCPUFreeRate := psCPU / estimateMaxPSCPU

	if len(psAvgCPU) > 1 {
		// If the overload of PS is not normal, the computation of  difference CPUn
		// will only be placed on one PS. Because variables are partitioned on PS by Round Robin.
		psCPUDiff := computePSCPUDiff(psAvgCPU)
		if psCPUDiff > 0 && estimatePSCPUFreeRate > psCPU/psCPUDiff {
			estimatePSCPUFreeRate = psCPU / psCPUDiff
		}
	}

	// Estimate the number of workers if the PS CPU is exhausted.
	estimateWorkerCount := math.Ceil(estimatePSCPUFreeRate * workerCount)
	if estimateWorkerCount < workerTargetReplica {
		workerTargetReplica = estimateWorkerCount
	}
	targetTotalPSCPU := (workerTargetReplica / workerCount) * maxSumUsedCPU

	psReplica := math.Ceil(targetTotalPSCPU / psCPU)

	psMemory := maxUsedMemory * (1 + memoryMarginPercent)

	resOptPlan := &common.AlgorithmOptimizePlan{
		JobRes: &common.JobResource{
			TaskGroupResources: map[string]*common.TaskGroupResource{
				common.PSTaskGroupName: {
					Count: int32(psReplica),
					Resource: &common.PodResource{
						Memory:  psMemory,
						CPUCore: float32(psCPU),
					},
				},
			},
		},
	}
	return resOptPlan, nil
}

func computePSCPUDiff(psAvgCPU map[uint64]float64) float64 {
	maxAvgCPU := 0.0
	maxPSNode := uint64(0)
	for ps, cpu := range psAvgCPU {
		if cpu > maxAvgCPU {
			maxAvgCPU = cpu
			maxPSNode = ps
		}
	}
	sumPSCPU := 0.0
	for ps, cpu := range psAvgCPU {
		if ps != maxPSNode {
			sumPSCPU += cpu
		}
	}
	if len(psAvgCPU) == 1 || sumPSCPU == 0.0 {
		return 0.0
	}
	avgPSCPU := sumPSCPU / float64(len(psAvgCPU)-1)
	CPUDiff := maxAvgCPU - avgPSCPU
	return CPUDiff
}

func getSumUsedCores(useds map[uint64]float64) float64 {
	sumUsedCPU := 0.0
	for _, used := range useds {
		sumUsedCPU += used
	}
	return sumUsedCPU
}

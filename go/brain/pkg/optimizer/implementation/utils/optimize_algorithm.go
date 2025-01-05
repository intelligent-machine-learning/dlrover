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

package utils

import (
	"encoding/json"
	log "github.com/golang/glog"
	"github.com/intelligent-machine-learning/easydl/brain/pkg/common"
	"github.com/intelligent-machine-learning/easydl/brain/pkg/config"
	optconfig "github.com/intelligent-machine-learning/easydl/brain/pkg/optimizer/config"
	optimplcomm "github.com/intelligent-machine-learning/easydl/brain/pkg/optimizer/implementation/common"
	"math"
	"reflect"
	"strconv"
)

// EstimateJobResourceByHistoricJobs estimates job resources according to historic jobs
func EstimateJobResourceByHistoricJobs(conf *optconfig.OptimizeAlgorithmConfig, jobMetrics *common.JobMetrics,
	historyJobs []*common.OptimizeJobMeta) (*common.AlgorithmOptimizePlan, error) {

	memoryMarginPercent, err := strconv.ParseFloat(conf.CustomizedConfig[config.OptimizerPSMemoryMarginPercent], 64)
	if err != nil || memoryMarginPercent == 0 {
		memoryMarginPercent = optimplcomm.DefaultPSMemoryMarginPercent
	}
	cpuMarginPercent, err := strconv.ParseFloat(conf.CustomizedConfig[config.OptimizerPSCPUMarginPercent], 64)
	if err != nil || cpuMarginPercent == 0 {
		cpuMarginPercent = optimplcomm.DefaultPSCPUMarginPercent
	}
	cpuMargin, err := strconv.ParseFloat(conf.CustomizedConfig[config.JobNodeCPUMargin], 64)
	if err != nil || cpuMargin == 0 {
		cpuMargin = optimplcomm.DefaultCPUMargin
	}
	maxPSCount, err := strconv.ParseFloat(conf.CustomizedConfig[config.OptimizerPSMaxCount], 64)
	if err != nil || cpuMargin == 0 {
		maxPSCount = optimplcomm.DefaultMaxPSCount
	}

	maxMemory := 0.0
	maxCPUCore := 0.0
	maxJobPSMemory := 0.0
	jobAvgPSCPUs := []float64{}

	for _, historyJob := range historyJobs {
		metrics := historyJob.Metrics

		rts := make([]*common.JobRuntimeInfo, 0)
		err = json.Unmarshal([]byte(metrics.JobRuntime), &rts)
		if err != nil {
			continue
		}

		l := len(rts)
		if l == 0 {
			continue
		}

		maxTotalPSMemory := 0.0
		taskPSCPU := make(map[uint64]float64)
		taskPSCPURecordNum := make(map[uint64]float64)
		CPUUsageSamples := make([]float64, 0)

		for _, rt := range rts {
			curTotalPSCPU := 0.0
			for n, cpu := range rt.PSCPU {
				taskPSCPU[n] += cpu
				taskPSCPURecordNum[n]++
				curTotalPSCPU += cpu
			}
			CPUUsageSamples = append(CPUUsageSamples, curTotalPSCPU)
		}
		majorCPUUsages := ComputeMajorCluster(CPUUsageSamples)
		jobAvgPSCPU := ComputeAverage(majorCPUUsages)
		jobAvgPSCPUs = append(jobAvgPSCPUs, jobAvgPSCPU)

		for n, totalCPU := range taskPSCPU {
			// Calculate the avg CPU of each node
			avgCPU := totalCPU / taskPSCPURecordNum[n]
			if maxCPUCore < avgCPU {
				maxCPUCore = avgCPU
			}
		}
		for _, rt := range rts {
			totalPSMemory := 0.0
			for _, memory := range rt.PSMemory {
				if maxMemory < memory {
					maxMemory = memory
				}
				totalPSMemory += memory
			}
			if maxTotalPSMemory < totalPSMemory {
				maxTotalPSMemory = totalPSMemory
			}
		}
		if maxJobPSMemory < maxTotalPSMemory {
			maxJobPSMemory = maxTotalPSMemory
			log.Infof("Job %s, total memory = %f, max memory = %f", metrics.JobName, maxJobPSMemory, maxMemory)
		}
	}

	majorJobCPUs := ComputeMajorCluster(jobAvgPSCPUs)
	avgJobPSCPU := ComputeAverage(majorJobCPUs)

	if avgJobPSCPU == 0 || maxMemory == 0 || maxCPUCore == 0 {
		return nil, nil
	}
	log.Errorf("Estimate the PS resource of job %s with the total CPU = %f, max CPU = %f", jobMetrics.JobName, avgJobPSCPU, maxCPUCore)

	cpuCore := maxCPUCore + cpuMargin
	totalCPUCore := avgJobPSCPU * (1 + cpuMarginPercent)
	replicaCount := math.Ceil(totalCPUCore / cpuCore)
	if replicaCount > maxPSCount {
		replicaCount = maxPSCount
		cpuCore = math.Ceil(totalCPUCore / replicaCount)
	}
	if maxMemory*replicaCount < maxJobPSMemory {
		maxMemory = math.Ceil(maxJobPSMemory / replicaCount)
	}
	memory := maxMemory * (1 + memoryMarginPercent)

	resOptPlan := &common.AlgorithmOptimizePlan{
		JobRes: &common.JobResource{
			TaskGroupResources: map[string]*common.TaskGroupResource{
				common.PSTaskGroupName: {
					Count: int32(replicaCount),
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

// ComputeAvgSpeed the average training speed
func ComputeAvgSpeed(runtimeInfos []*common.JobRuntimeInfo, maxCount int) float64 {
	l := len(runtimeInfos)

	count := minInt(l, maxCount)
	if count == 0 {
		return 0
	}

	totalSpeed := float64(0)

	for i := l - 1; i >= l-count; i-- {
		totalSpeed = totalSpeed + runtimeInfos[i].Speed/float64(len(runtimeInfos[i].WorkerCPU))
	}
	return totalSpeed / float64(count)
}

// ComputePerStepTime computes time of per step of training
func ComputePerStepTime(metrics *common.JobMetrics, speed float64) (float64, error) {
	trainingHyperParam := &common.TrainingHyperParams{}
	err := json.Unmarshal([]byte(metrics.HyperParamsFeature), trainingHyperParam)
	if err != nil {
		log.Errorf("Fail to unmarshal hyper parameters: %v", err)
		return 0, err
	}
	batchSize := float64(trainingHyperParam.BatchSize)

	trainingDatasetFeature := &common.TrainingSetFeature{}
	err = json.Unmarshal([]byte(metrics.DatasetFeature), trainingDatasetFeature)
	if err != nil {
		log.Errorf("Fail to unmarshal dataset feature: %v", err)
		return 0, err
	}
	datasetSize := float64(trainingDatasetFeature.DatasetSize)

	steps := datasetSize / batchSize
	epoch := float64(trainingHyperParam.Epoch)
	maxSteps := float64(trainingHyperParam.MaxSteps)
	if epoch > 0 {
		steps = epoch * steps
	}
	if maxSteps > 0 && steps > maxSteps {
		steps = maxSteps
	}

	time := steps / speed
	return time, nil
}

func minInt(x, y int) int {
	if x < y {
		return x
	}
	return y
}

// FilterRuntimeInfosWithLatestPS get the runtime infos with the latest PS group
func FilterRuntimeInfosWithLatestPS(runtimeInfos []*common.JobRuntimeInfo) []*common.JobRuntimeInfo {
	newInfos := make([]*common.JobRuntimeInfo, 0)

	l := len(runtimeInfos)
	lastPsIds := make(map[uint64]bool)

	if l > 0 {
		lastInfo := runtimeInfos[l-1]
		for i := range lastInfo.PSCPU {
			lastPsIds[i] = true
		}
	}
	for _, info := range runtimeInfos {
		psIds := make(map[uint64]bool)
		for i := range info.PSCPU {
			psIds[i] = true
		}
		// Filter invalid records where the PS set is not equal the last set.
		if !reflect.DeepEqual(lastPsIds, psIds) {
			continue
		}
		newInfos = append(newInfos, info)
	}
	return newInfos
}

// CheckHotCPUNodes check whether there is node whose CPU util is beyond hotThreshold
func CheckHotCPUNodes(runtimeInfos []*common.JobRuntimeInfo, nodeCPUs map[uint64]float64, hotThreshold float64, sampleStep int) []uint64 {
	hotPsIds := make([]uint64, 0)
	if len(runtimeInfos) < sampleStep {
		return hotPsIds
	}
	psAvgCPU := CalculateJobNodeAvgResources(runtimeInfos, sampleStep, optimplcomm.ResourceTypePSCPU)
	for n, cpu := range psAvgCPU {
		totalCPU, found := nodeCPUs[n]
		if !found {
			log.Errorf("fail to find node %d total cpu", n)
			continue
		}
		cpuUtil := cpu / totalCPU
		if cpuUtil > hotThreshold {
			hotPsIds = append(hotPsIds, n)
			log.Errorf("The CPU util %f of PS %d is overload %f ", cpuUtil, n, hotThreshold)
		}
	}

	return hotPsIds
}

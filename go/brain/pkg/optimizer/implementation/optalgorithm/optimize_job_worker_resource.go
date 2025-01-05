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
	"reflect"
	"strconv"
)

const (
	// OptimizeAlgorithmJobWorkerResource is the name of worker resource optimize function
	OptimizeAlgorithmJobWorkerResource = "optimize_job_worker_resource"
	defaultEnoughRecordNum             = 3
	speedDecelerated                   = "decelerated"
	speedIncreased                     = "increased"
	speedStable                        = "stable"
	initTrainingRecordNumThres         = 6
	maxWorkerIncreasedMemory           = 8 * 1024 * 1024 * 1024
)

func init() {
	registerOptimizeAlgorithm(OptimizeAlgorithmJobWorkerResource, OptimizeJobWorkerResource)
}

// OptimizeJobWorkerResource optimizes worker resource at runtime
func OptimizeJobWorkerResource(dataStore datastoreapi.DataStore, conf *optconfig.OptimizeAlgorithmConfig, optJob *common.OptimizeJobMeta,
	historyJobs []*common.OptimizeJobMeta) (*common.AlgorithmOptimizePlan, error) {

	metrics := optJob.Metrics
	runtimeInfos := make([]*common.JobRuntimeInfo, 0)
	err := json.Unmarshal([]byte(metrics.JobRuntime), &runtimeInfos)
	if err != nil {
		log.Errorf("Fail to unmarshal runtime info: %v", err)
		return nil, err
	}

	if len(runtimeInfos) == 0 {
		log.Info("There is no runtime infos")
		return nil, nil
	}

	psTasks := optimplutils.GetJobNodesByGroup(dataStore, optJob.JobMeta, common.PSTaskGroupName)
	psCPUs := optimplutils.GetResourceFromJobNode(psTasks, optimplcomm.ResourceTypeCPU)

	maxReplicaCount, err := strconv.Atoi(conf.CustomizedConfig[config.OptimizerWorkerMaxReplicaCount])
	if err != nil {
		log.Errorf("Fail to get cpu util compare count %s: %v", conf.CustomizedConfig[config.OptimizerWorkerMaxReplicaCount], err)
		return nil, err
	}

	cpuUtilCompCount, err := strconv.Atoi(conf.CustomizedConfig[config.OptimizerWorkerCPUUtilCompCount])
	if err != nil {
		log.Errorf("Fail to get cpu util compare count %s: %v", conf.CustomizedConfig[config.OptimizerWorkerCPUUtilCompCount], err)
		return nil, err
	}
	cpuUtilLessPercent, err := strconv.ParseFloat(conf.CustomizedConfig[config.OptimizerWorkerCPUUtilLessPercent], 64)
	if err != nil {
		log.Errorf("Fail to get cpu util less percent %s: %v", conf.CustomizedConfig[config.OptimizerWorkerCPUUtilLessPercent], err)
		return nil, err
	}

	stepCountThreshold, err := strconv.Atoi(conf.CustomizedConfig[config.OptimizerStepCountThreshold])
	if err != nil {
		log.Errorf("Fail to get step count threshold %s: %v", conf.CustomizedConfig[config.OptimizerStepCountThreshold], err)
		return nil, err
	}
	speedLessPercent, err := strconv.ParseFloat(conf.CustomizedConfig[config.OptimizerTrainingSpeedLessPercent], 64)
	if err != nil {
		log.Errorf("Fail to get training speed less percent %s: %v", conf.CustomizedConfig[config.OptimizerTrainingSpeedLessPercent], err)
		return nil, err
	}
	workerReplicaDecreaseCount, err := strconv.Atoi(conf.CustomizedConfig[config.OptimizerWorkerReplicaDecreaseCount])
	if err != nil {
		log.Errorf("Fail to get worker replica decrease count %s: %v", conf.CustomizedConfig[config.OptimizerWorkerReplicaDecreaseCount], err)
		return nil, err
	}
	psCPUOverload, err := strconv.ParseFloat(conf.CustomizedConfig[config.OptimizerPSCPUOverload], 64)
	if err != nil {
		log.Errorf("Fail to get ps cpu overload %s: %v", conf.CustomizedConfig[config.OptimizerPSCPUOverload], err)
		return nil, err
	}
	psCPUExhaustedThreshold, err := strconv.ParseFloat(conf.CustomizedConfig[config.OptimizerPSCPUExhaustedThreshold], 64)
	if err != nil {
		log.Errorf("Fail to get ps cpu overload %s: %v", conf.CustomizedConfig[config.OptimizerPSCPUExhaustedThreshold], err)
		return nil, err
	}
	workerMaxInitCountPerStep, err := strconv.Atoi(conf.CustomizedConfig[config.OptimizerWorkerMaxInitCountPerStep])
	if err != nil {
		log.Errorf("Fail to get worker max init count per step %s: %v", conf.CustomizedConfig[config.OptimizerWorkerMaxInitCountPerStep], err)
		return nil, err
	}
	workerMaxCountPerStep, err := strconv.Atoi(conf.CustomizedConfig[config.OptimizerWorkerMaxCountPerStep])
	if err != nil {
		log.Errorf("Fail to get worker max count per step %s: %v", conf.CustomizedConfig[config.OptimizerWorkerMaxCountPerStep], err)
		return nil, err
	}
	workerMemoryMarginPercent, err := strconv.ParseFloat(conf.CustomizedConfig[config.OptimizerWorkerMemoryMarginPercent], 64)
	if err != nil {
		log.Errorf("Fail to get worker memory margin percent %s: %v", conf.CustomizedConfig[config.OptimizerWorkerMemoryMarginPercent], err)
		return nil, err
	}
	workerCPUMarginCores, err := strconv.ParseFloat(conf.CustomizedConfig[config.OptimizerWorkerCPUMarginCore], 64)
	if err != nil {
		log.Errorf("Fail to get worker cpu margin core %s: %v", conf.CustomizedConfig[config.OptimizerWorkerCPUMarginCore], err)
		return nil, err
	}

	runtimeInfos = preProcessRuntimeInfos(runtimeInfos, psCPUs, psCPUOverload, cpuUtilCompCount, cpuUtilLessPercent)
	recordNum := len(runtimeInfos)
	if recordNum < cpuUtilCompCount {
		log.Errorf("No enough records! The number of records is %d", recordNum)
		return nil, nil
	}

	latestRuntimeInfo := runtimeInfos[recordNum-1]
	replica := len(latestRuntimeInfo.WorkerCPU)
	currWorkerReplica := len(latestRuntimeInfo.WorkerCPU)

	psMaxCPU := optimplutils.CalculateJobNodeMaxResource(runtimeInfos, optimplcomm.NRecordToAvgResource, optimplcomm.ResourceTypePSCPU)
	maxPSCPUUtil := optimplutils.GetMaxUtil(psMaxCPU, psCPUs)

	speedState := getTrainingSpeedState(runtimeInfos, stepCountThreshold, speedLessPercent)
	log.Infof("Job: %s, maxPSCPUUtil = %f, speed state = %s", metrics.JobName, maxPSCPUUtil, speedState)
	exhaustedPSNodes := optimplutils.CheckHotCPUNodes(runtimeInfos, psCPUs, psCPUExhaustedThreshold, defaultEnoughRecordNum)
	// Decrease the number of worker if there is exhausted PS node
	if len(exhaustedPSNodes) > 0 {
		if replica > workerReplicaDecreaseCount {
			replica = replica - workerReplicaDecreaseCount
		}
	} else if maxPSCPUUtil < psCPUOverload && speedState != speedDecelerated {
		if maxPSCPUUtil <= 0.0 {
			replica = replica + workerMaxCountPerStep
		} else {
			replica = computeWorkerReplicaForIdlePSCPU(maxPSCPUUtil, psCPUOverload, currWorkerReplica)
		}

		phase := conf.CustomizedConfig[config.OptimizerWorkerOptimizePhase]
		if phase == config.OptimizerWorkerOptimizePhaseInitial || phase == config.OptimizerWorkerOptimizePhaseSample {
			avgSpeed := computePreAvgSpeed(runtimeInfos, stepCountThreshold)
			if avgSpeed == float64(0) {
				log.Warningf("Fail to compute job %s initial speed", optJob.JobMeta.Name)
				replica = currWorkerReplica + min(workerMaxCountPerStep, replica-currWorkerReplica)
			} else {
				time, err := computeInitPerStepTime(metrics, avgSpeed)
				if err != nil {
					log.Errorf("Fail to compute job %s initial step time: %v", optJob.JobMeta.Name, err)
				} else if time <= initStepTime {
					replica = defaultInitWorker
				} else {
					replica = min(workerMaxInitCountPerStep, replica)
				}
			}
		} else if phase == config.OptimizerWorkerOptimizePhaseStable && speedState == speedIncreased {
			replica = currWorkerReplica + min(workerMaxCountPerStep, replica-currWorkerReplica)
		} else {
			log.Errorf("invalid optimize phase: %s", phase)
		}
	}

	memory := 0.0
	cpuCore := 0.0

	var workerCPU map[uint64]float64
	if len(runtimeInfos) < initTrainingRecordNumThres {
		// In the startup of training, the CPU usage of workers is not stable.
		// We use the maximum of used CPU cores to estimate the CPU.
		workerCPU = optimplutils.CalculateJobNodeMaxResource(runtimeInfos, optimplcomm.NRecordToAvgResource, optimplcomm.ResourceTypeWorkerCPU)
	} else {
		workerCPU = optimplutils.CalculateJobNodeAvgResources(runtimeInfos, optimplcomm.NRecordToAvgResource, optimplcomm.ResourceTypeWorkerCPU)
	}
	for _, cpu := range workerCPU {
		if cpu > cpuCore {
			cpuCore = cpu
		}
	}
	for _, runtimeInfo := range runtimeInfos {
		for _, mem := range runtimeInfo.WorkerMemory {
			if mem > memory {
				memory = mem
			}
		}
	}

	addMemory := memory * workerMemoryMarginPercent
	if addMemory > maxWorkerIncreasedMemory {
		addMemory = maxWorkerIncreasedMemory
	}
	memory = memory + addMemory
	if cpuCore > 0.0 {
		// cpuCore = 0.0 is an invalid optimization result
		cpuCore = math.Ceil(cpuCore + workerCPUMarginCores)
	}

	if replica > maxReplicaCount {
		replica = maxReplicaCount
	}

	resOptPlan := &common.AlgorithmOptimizePlan{
		JobRes: &common.JobResource{
			TaskGroupResources: map[string]*common.TaskGroupResource{
				common.WorkerTaskGroupName: {
					Count: int32(replica),
					Resource: &common.PodResource{
						CPUCore: float32(cpuCore),
						Memory:  memory,
						GPUCore: float32(0.0),
						GPUType: "",
					},
				},
			},
		},
	}
	return resOptPlan, nil
}

func computeWorkerReplicaForIdlePSCPU(psCPU float64, psCPUOverload float64, replica int) int {
	if psCPU >= psCPUOverload {
		return replica
	}
	return int(math.Ceil((psCPUOverload / psCPU) * float64(replica)))
}

func getTrainingSpeedState(infos []*common.JobRuntimeInfo, count int, lessPercent float64) string {
	l := len(infos)
	curReplica := 0
	lastReplicaInter := -1
	for i := l - 1; i >= 0; i-- {
		info := infos[i]
		if curReplica == 0 {
			curReplica = len(info.WorkerCPU)
		} else if curReplica != len(info.WorkerCPU) {
			lastReplicaInter = i
			break
		}
	}

	if lastReplicaInter > (l - count - 1) {
		log.Infof("Not enough runtime info")
		return speedStable
	} else if lastReplicaInter < (count - 1) {
		return speedIncreased
	}

	preTotalSpeed := float64(0)
	posTotalSpeed := float64(0)
	for i := lastReplicaInter; i >= lastReplicaInter-count+1; i-- {
		preTotalSpeed = preTotalSpeed + infos[i].Speed
	}
	for i := lastReplicaInter + 1; i <= lastReplicaInter+count; i++ {
		posTotalSpeed = posTotalSpeed + infos[i].Speed
	}

	preAvgSpeed := preTotalSpeed / float64(count)
	posAvgSpeed := posTotalSpeed / float64(count)
	if speedLessThan(preAvgSpeed, posAvgSpeed, lessPercent) {
		return speedDecelerated
	} else if preAvgSpeed < posAvgSpeed {
		return speedIncreased
	} else {
		return speedStable
	}
}

func speedLessThan(preSpeed float64, posSpeed float64, lessPercent float64) bool {
	return preSpeed > posSpeed && (preSpeed-posSpeed)/preSpeed >= lessPercent
}

func computeInitPerStepTime(metrics *common.JobMetrics, speed float64) (float64, error) {
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
	if maxSteps > 0 {
		if steps > maxSteps {
			steps = maxSteps
		}
	}

	time := steps / speed
	return time, nil
}

func computePreAvgSpeed(runtimeInfos []*common.JobRuntimeInfo, maxCount int) float64 {
	l := len(runtimeInfos)

	count := min(l, maxCount)
	if count == 0 {
		return 0
	}

	totalSpeed := float64(0)

	for i := l - 1; i >= l-count; i-- {
		totalSpeed = totalSpeed + runtimeInfos[i].Speed/float64(len(runtimeInfos[i].WorkerCPU))
	}
	return totalSpeed / float64(count)
}

func min(x, y int) int {
	if x < y {
		return x
	}
	return y
}

func preProcessRuntimeInfos(runtimeInfos []*common.JobRuntimeInfo, cpus map[uint64]float64, overloadCPUUtil float64, count int, lessPercent float64) []*common.JobRuntimeInfo {
	newInfos := make([]*common.JobRuntimeInfo, 0)

	l := len(runtimeInfos)
	lastPsIds := make(map[uint64]bool)

	if l > 0 {
		lastInfo := runtimeInfos[l-1]
		for i := range lastInfo.PSCPU {
			lastPsIds[i] = true
		}
	}
	validInfoNum := 0
	for i, info := range runtimeInfos {
		psIds := make(map[uint64]bool)
		for i := range info.PSCPU {
			psIds[i] = true
		}
		// Filter invalid records where the PS set is not equal the last set.
		if !reflect.DeepEqual(lastPsIds, psIds) {
			continue
		}

		if validInfoNum == 0 || i == l-1 {
			newInfos = append(newInfos, info)
			validInfoNum++
			continue
		}

		isSingularity := true
		cpuUtil := optimplutils.GetMaxUtil(info.PSCPU, cpus)
		if cpuUtil <= overloadCPUUtil {
			newInfos = append(newInfos, info)
			validInfoNum++
			continue
		}

		for j := i - count; j <= i+count; j++ {
			if j < 0 || j == i || j >= l {
				continue
			}

			compInfo := runtimeInfos[j]
			compCPUUtil := optimplutils.GetMaxUtil(compInfo.PSCPU, cpus)
			if cpuUtil <= compCPUUtil || (cpuUtil-compCPUUtil)/cpuUtil < lessPercent {
				isSingularity = false
				break
			}
		}

		if !isSingularity {
			newInfos = append(newInfos, info)
			validInfoNum++
		}
	}
	return newInfos
}

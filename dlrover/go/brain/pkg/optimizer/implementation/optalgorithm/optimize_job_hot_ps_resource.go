package optalgorithm

import (
	"github.com/intelligent-machine-learning/easydl/brain/pkg/common"
	datastoreapi "github.com/intelligent-machine-learning/easydl/brain/pkg/datastore/api"
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
func OptimizeJobHotPSResource(dataStore datastoreapi.DataStore, conf *optimizerconfig.OptimizeAlgorithmConfig, optJob *common.OptimizeJobMeta,
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
		Type: common.TypeGetDataListTask,
		Extra: &cougardb.TaskCondition{
			JobUUID:       utils.GetJobUUIDForDBRecord(optJob.JobMeta.UUID),
			TaskGroupName: common.PSTaskGroupName,
		},
	}
	tasks := make([]*cougardb.Task, 0)
	err = dataStore.GetData(cond, &tasks)
	if err != nil {
		log.Errorf("fail to get tasks for %s: %v", optJob.JobMeta.UUID, err)
		return nil, err
	}

	taskRes := make(map[uint64]float64)
	taskMemory := make(map[uint64]float64)
	taskNames := make(map[uint64]string)
	for _, task := range tasks {
		num, err := utils.ExtractTaskNumberFromName(task.Name)
		if err != nil {
			log.Errorf("fail to extract number for %s", task.Name)
			continue
		}
		taskRes[num] = float64(task.CPU)
		taskMemory[num] = float64(task.Memory)
		taskNames[num] = task.Name
	}

	runtimeInfos := make([]*common.JobRuntimeInfo, 0)
	err = json.Unmarshal([]byte(optJob.Metrics.MetricJobRuntime), &runtimeInfos)
	if err != nil {
		log.Errorf("fail to unmarshal runtime info for %s: %v", optJob.JobMeta.Name, err)
		return nil, err
	}

	if len(runtimeInfos) == 0 {
		log.Info("there is no runtime infos")
		return nil, nil
	}

	runtimeInfos = implutils.FilterRuntimeInfosWithLatestPS(runtimeInfos)

	optTaskRes := make(map[string]*common.PodResource)

	hotCPUPsNodes := implutils.CheckHotCPUNodes(
		runtimeInfos, taskRes, hotCPUThreshold, implutils.RecordNumToAvgResource)
	hotMemoryPsNodes := checkHotMemoryNodes(
		runtimeInfos, taskMemory, hotMemoryThreshold, implutils.RecordNumToAvgResource)

	if len(hotCPUPsNodes) > 0 {
		rt := runtimeInfos[len(runtimeInfos)-1]
		curWorkerNum := len(rt.WorkerCPU)
		avgCPU := implutils.CalculatePSAvgCPU(runtimeInfos, implutils.RecordNumToAvgResource)

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
			if optCPU > taskRes[n] {
				taskName := taskNames[n]
				optTaskRes[taskName] = &common.PodResource{
					CPUCore: float32(optCPU),
				}
			}
		}
	}

	for _, n := range hotMemoryPsNodes {
		totalMemory, found := taskMemory[n]
		if !found {
			log.Errorf("fail to find task %d total cpu", n)
			continue
		}
		taskName := taskNames[n]
		optMemory := totalMemory + float64(hotAdjustMemory)
		if _, ok := optTaskRes[taskName]; ok {
			optTaskRes[taskName].Memory = optMemory
		} else {
			optTaskRes[taskName] = &common.PodResource{
				Memory: optMemory,
			}
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

func checkHotMemoryNodes(runtimeInfos []*common.JobRuntimeInfo, taskMemory map[uint64]float64, hotThreshold float64, checkHotStep int) []uint64 {
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
			totalMemory, found := taskMemory[n]
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

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
	"fmt"
	log "github.com/golang/glog"
	"github.com/intelligent-machine-learning/easydl/brain/pkg/common"
	datastoreapi "github.com/intelligent-machine-learning/easydl/brain/pkg/datastore/api"
	"github.com/intelligent-machine-learning/easydl/brain/pkg/datastore/recorder/mysql"
	pb "github.com/intelligent-machine-learning/easydl/brain/pkg/proto"
	"github.com/intelligent-machine-learning/easydl/brain/pkg/utils"
)

const (
	// MaxRuntimeInfoRecords is the max number of runtime info records
	MaxRuntimeInfoRecords = 60
)

// GetData returns data for a given condition
func GetData(client *mysql.Client, condition *datastoreapi.Condition, data interface{}) error {
	switch condition.Type {
	case common.TypeGetDataGetJobMetrics:
		_, ok := condition.Extra.(*mysql.JobMetricsCondition)
		if !ok {
			return fmt.Errorf("GetData %s condition is not *JobMetricsCondition", condition.Type)
		}
		_, ok = data.(*mysql.JobMetrics)
		if !ok {
			return fmt.Errorf("GetData %s data is not *JobMetrics", condition.Type)
		}
		return client.JobMetricsRecorder.Get(condition.Extra.(*mysql.JobMetricsCondition), data.(*mysql.JobMetrics))
	case common.TypeGetDataListJobMetrics:
		_, ok := condition.Extra.(*mysql.JobMetricsCondition)
		if !ok {
			return fmt.Errorf("GetData %s condition is not *JobMetricsCondition", condition.Type)
		}
		_, ok = data.(*[]*mysql.JobMetrics)
		if !ok {
			return fmt.Errorf("GetData %s data is not *[]*JobMetrics", condition.Type)
		}
		return client.JobMetricsRecorder.List(condition.Extra.(*mysql.JobMetricsCondition), data.(*[]*mysql.JobMetrics))
	case common.TypeGetDataGetJob:
		_, ok := condition.Extra.(*mysql.JobCondition)
		if !ok {
			return fmt.Errorf("GetData %s condition is not *JobCondition", condition.Type)
		}
		_, ok = data.(*mysql.Job)
		if !ok {
			return fmt.Errorf("GetData %s data is not *Job", condition.Type)
		}
		return client.JobRecorder.Get(condition.Extra.(*mysql.JobCondition), data.(*mysql.Job))
	case common.TypeGetDataListJob:
		_, ok := condition.Extra.(*mysql.JobCondition)
		if !ok {
			return fmt.Errorf("GetData %s condition is not *JobCondition", condition.Type)
		}
		_, ok = data.(*[]*mysql.Job)
		if !ok {
			return fmt.Errorf("GetData %s data is not *[]*Job", condition.Type)
		}
		return client.JobRecorder.List(condition.Extra.(*mysql.JobCondition), data.(*[]*mysql.Job))
	case common.TypeGetDataGetJobNode:
		_, ok := condition.Extra.(*mysql.JobNodeCondition)
		if !ok {
			return fmt.Errorf("GetData %s condition is not *JobNodeCondition", condition.Type)
		}
		_, ok = data.(*mysql.JobNode)
		if !ok {
			return fmt.Errorf("GetData %s data is not *JobNode", condition.Type)
		}
		return client.JobNodeRecorder.Get(condition.Extra.(*mysql.JobNodeCondition), data.(*mysql.JobNode))
	case common.TypeGetDataListJobNode:
		_, ok := condition.Extra.(*mysql.JobNodeCondition)
		if !ok {
			return fmt.Errorf("GetData %s condition is not *JobNodeCondition", condition.Type)
		}
		_, ok = data.(*[]*mysql.JobNode)
		if !ok {
			return fmt.Errorf("GetData %s data is not *[]*JobNode", condition.Type)
		}
		return client.JobNodeRecorder.List(condition.Extra.(*mysql.JobNodeCondition), data.(*[]*mysql.JobNode))
	}

	return fmt.Errorf("invalid type: %s", condition.Type)
}

// PersistJobMetrics persists job metrics record into mysql db
func PersistJobMetrics(client *mysql.Client, jobMetrics *pb.JobMetrics) error {
	storeJobMetrics := queryJobMetrics(client, jobMetrics.JobMeta.Uuid)
	log.Infof("Report job metrics is %s", jobMetrics.JobMeta)

	switch jobMetrics.MetricsType {
	case pb.MetricsType_Job_Exit_Reason:
		return persistExitReason(client, storeJobMetrics, jobMetrics.GetJobExitReason())
	case pb.MetricsType_Model_Feature:
		in := jobMetrics.GetModelFeature()
		modelFeature := &common.ModelFeature{
			VariableCount:      uint64(in.GetVariableCount()),
			OpCount:            uint64(in.GetOpCount()),
			EmbeddingDimension: uint64(in.GetEmbeddingDimension()),
			TotalVariableSize:  uint64(in.GetTotalVariableSize()),
			MaxVariableSize:    uint64(in.GetMaxVariableSize()),
			UpdateOpCount:      uint64(in.GetUpdateOpCount()),
			ReadOpCount:        uint64(in.GetReadOpCount()),
			InputFetchDur:      uint64(in.GetInputFetchDur()),
			Flops:              uint64(in.GetFlops()),
			RecvOpCount:        uint64(in.GetRecvOpCount()),
			KvEmbeddingDims:    in.GetKvEmbeddingDims(),
			TensorAllocBytes:   in.GetTensorAllocBytes(),
		}
		return persistModelFeature(client, storeJobMetrics, modelFeature)
	case pb.MetricsType_Runtime_Info:
		return persistRuntimeInfo(client, storeJobMetrics, jobMetrics.GetRuntimeInfo())
	case pb.MetricsType_Training_Hyper_Params:
		in := jobMetrics.GetTrainingHyperParams()
		trainingHyperParams := &common.TrainingHyperParams{
			BatchSize: uint64(in.GetBatchSize()),
			Epoch:     int(in.GetEpoch()),
			MaxSteps:  uint64(in.GetMaxSteps()),
		}
		return persistTrainingHyperParams(client, storeJobMetrics, trainingHyperParams)
	case pb.MetricsType_Workflow_Feature:
		in := jobMetrics.GetWorkflowFeature()
		workflowFeature := &common.WorkflowFeature{
			UserID:      in.GetUserId(),
			JobName:     in.GetJobName(),
			CodeAddress: in.GetCodeAddress(),
			WorkflowID:  in.GetWorkflowId(),
			NodeID:      in.GetNodeId(),
			OdpsProject: in.GetOdpsProject(),
			IsProd:      in.GetIsProd(),
		}
		return persistWorkflowFeature(client, storeJobMetrics, workflowFeature)
	case pb.MetricsType_Training_Set_Feature:
		in := jobMetrics.GetTrainingSetFeature()
		trainingSetFeature := &common.TrainingSetFeature{
			DatasetSize:         uint64(in.GetDatasetSize()),
			DatasetName:         in.GetDatasetName(),
			SparseItemCount:     uint64(in.GetSparseItemCount()),
			SparseFeatures:      in.GetSparseFeatures(),
			SparseFeatureGroups: in.GetSparseFeatureGroups(),
			SparseFeatureShapes: in.GetSparseFeatureShapes(),
			DenseFeatures:       in.GetDenseFeatures(),
			DenseFeatureShapes:  in.GetDenseFeatureShapes(),
			StorageSize:         uint64(in.GetStorageSize()),
		}
		return persistTrainingSetFeature(client, storeJobMetrics, trainingSetFeature)
	case pb.MetricsType_Type:
		return persistType(client, storeJobMetrics, jobMetrics.GetType())
	case pb.MetricsType_Resource:
		return persistResource(client, storeJobMetrics, jobMetrics.GetResource())
	case pb.MetricsType_Customized_Data:
		return persistCustomizedData(client, storeJobMetrics, jobMetrics.GetCustomizedData())
	case pb.MetricsType_Optimization:
		return persistOptimization(client, storeJobMetrics, jobMetrics.GetJobOptimization())
	}
	return fmt.Errorf("invalid JobMetrics type: %s", jobMetrics.MetricsType)
}

func queryJobMetrics(client *mysql.Client, jobUUID string) *mysql.JobMetrics {
	cond := &mysql.JobMetricsCondition{
		UID: jobUUID,
	}
	jobMetrics := &mysql.JobMetrics{}
	err := client.JobMetricsRecorder.Get(cond, jobMetrics)
	if err != nil {
		log.Infof("Not found job %s in db, insert a new one.", jobUUID)
		jobMetrics.UID = jobUUID
	}
	return jobMetrics
}

func persistTrainingHyperParams(client *mysql.Client, jobMetrics *mysql.JobMetrics, params *common.TrainingHyperParams) error {
	hyperParams := &common.TrainingHyperParams{}
	err := json.Unmarshal([]byte(jobMetrics.HyperParamsFeature), hyperParams)
	if err != nil {
		hyperParams = params
	} else {
		// DLRover may update training hyper parameters during training
		hyperParams.Update(params)
	}
	jsonVal, _ := json.Marshal(hyperParams)
	jobMetrics.HyperParamsFeature = string(jsonVal)
	err = client.JobMetricsRecorder.Upsert(jobMetrics)
	return err
}

func persistWorkflowFeature(client *mysql.Client, jobMetrics *mysql.JobMetrics, params *common.WorkflowFeature) error {
	jsonVal, _ := json.Marshal(params)
	jobMetrics.JobFeature = string(jsonVal)
	err := client.JobMetricsRecorder.Upsert(jobMetrics)
	return err
}

func persistTrainingSetFeature(client *mysql.Client, jobMetrics *mysql.JobMetrics, params *common.TrainingSetFeature) error {
	trainingSet := &common.TrainingSetFeature{}
	err := json.Unmarshal([]byte(jobMetrics.DatasetFeature), trainingSet)
	if err != nil {
		trainingSet = params
	} else {
		// DLRover may update training dataset metrics during training
		trainingSet.Update(params)
	}
	jsonVal, _ := json.Marshal(trainingSet)
	jobMetrics.DatasetFeature = string(jsonVal)
	err = client.JobMetricsRecorder.Upsert(jobMetrics)
	return err
}

func persistModelFeature(client *mysql.Client, jobMetrics *mysql.JobMetrics, params *common.ModelFeature) error {
	jsonVal, _ := json.Marshal(params)
	jobMetrics.ModelFeature = string(jsonVal)
	err := client.JobMetricsRecorder.Upsert(jobMetrics)
	return err
}

func persistRuntimeInfo(client *mysql.Client, jobMetrics *mysql.JobMetrics, in *pb.RuntimeInfo) error {
	runningPods := in.GetRunningPods()

	workerMemory := make(map[uint64]float64)
	workerCPU := make(map[uint64]float64)
	psMemory := make(map[uint64]float64)
	psCPU := make(map[uint64]float64)

	for _, pod := range runningPods {
		podType, id := utils.ExtractPodTypeAndIDFromName(pod.PodName)
		if id < 0 {
			log.Errorf("invalid pod name: %s", pod.PodName)
			continue
		}

		podID := uint64(id)
		if podType == common.WorkerTaskGroupName {
			workerMemory[podID] = pod.MemUsage
			workerCPU[podID] = pod.CpuUsage
		} else if podType == common.PSTaskGroupName {
			psMemory[podID] = pod.MemUsage
			psCPU[podID] = pod.CpuUsage
		}
	}

	speed := utils.Decimal(float64(in.GetSpeed()))
	jobRuntimeInfo := &common.JobRuntimeInfo{
		GlobalStep:   uint64(in.GetGlobalStep()),
		TimeStamp:    uint64(in.GetTimeStamp()),
		Speed:        speed,
		WorkerMemory: workerMemory,
		WorkerCPU:    workerCPU,
		PSMemory:     psMemory,
		PSCPU:        psCPU,
	}

	var runtimeInfos []*common.JobRuntimeInfo

	if jobMetrics.JobRuntime != "" {
		err := json.Unmarshal([]byte(jobMetrics.JobRuntime), &runtimeInfos)
		if err != nil {
			log.Errorf("Fail to unmarshal runtime info: %v", err)
			return err
		}
	}
	runtimeInfos = append(runtimeInfos, jobRuntimeInfo)
	if len(runtimeInfos) > MaxRuntimeInfoRecords {
		runtimeInfos = runtimeInfos[1:]
	}
	jsonVal, err := json.Marshal(runtimeInfos)
	if err != nil {
		log.Errorf("Fail to dump json %v", err)
	}
	jobMetrics.JobRuntime = string(jsonVal)
	err = client.JobMetricsRecorder.Upsert(jobMetrics)
	return err
}

func persistExitReason(client *mysql.Client, jobMetrics *mysql.JobMetrics, reason string) error {
	jobMetrics.ExitReason = reason
	err := client.JobMetricsRecorder.Upsert(jobMetrics)
	return err
}

func persistOptimization(client *mysql.Client, jobMetrics *mysql.JobMetrics, in *pb.JobOptimization) error {
	var jobOptimizations []*pb.JobOptimization
	if jobMetrics.Optimization != "" {
		err := json.Unmarshal([]byte(jobMetrics.Optimization), &jobOptimizations)
		if err != nil {
			log.Errorf("Fail to unmarshal %s optimization json: %v", jobMetrics.UID, err)
		}
	}

	if len(jobOptimizations) == 0 || in.Plan != nil {
		jobOptimizations = append(jobOptimizations, in)
	} else {
		l := len(jobOptimizations)
		jobOpt := jobOptimizations[l-1]
		for _, jobState := range in.JobStates {
			jobOpt.JobStates = append(jobOpt.JobStates, jobState)
		}
	}

	optimizationVal, err := json.Marshal(jobOptimizations)
	if err != nil {
		log.Errorf("Fail to dump %s optimization json: %v", jobMetrics.UID, err)
	}

	jobMetrics.Optimization = string(optimizationVal)
	err = client.JobMetricsRecorder.Upsert(jobMetrics)
	return err
}

func persistType(client *mysql.Client, jobMetrics *mysql.JobMetrics, tp string) error {
	jobMetrics.Type = tp
	err := client.JobMetricsRecorder.Upsert(jobMetrics)
	return err
}

func persistResource(client *mysql.Client, jobMetrics *mysql.JobMetrics, resource string) error {
	jobMetrics.Resource = resource
	err := client.JobMetricsRecorder.Upsert(jobMetrics)
	return err
}

func persistCustomizedData(client *mysql.Client, jobMetrics *mysql.JobMetrics, data string) error {
	jobMetrics.CustomizedData = data
	err := client.JobMetricsRecorder.Upsert(jobMetrics)
	return err
}

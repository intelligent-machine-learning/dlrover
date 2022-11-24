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

package implementation

import (
	"encoding/json"
	"fmt"
	log "github.com/golang/glog"
	"github.com/intelligent-machine-learning/easydl/brain/pkg/common"
	"github.com/intelligent-machine-learning/easydl/brain/pkg/config"
	datastoreapi "github.com/intelligent-machine-learning/easydl/brain/pkg/datastore/api"
	dsimplutils "github.com/intelligent-machine-learning/easydl/brain/pkg/datastore/implementation/utils"
	"github.com/intelligent-machine-learning/easydl/brain/pkg/datastore/recorder/mysql"
	pb "github.com/intelligent-machine-learning/easydl/brain/pkg/proto"
	"github.com/intelligent-machine-learning/easydl/brain/pkg/utils"
	"time"
)

const (
	// MaxRuntimeInfoRecords is the max number of runtime info records
	MaxRuntimeInfoRecords = 60
	// BaseDataStoreName is the name of base data store
	BaseDataStoreName = "base_datastore"
)

func init() {
	registerNewFunc(BaseDataStoreName, newBaseDataStore)
}

// BaseDataStore is the base data store
type BaseDataStore struct {
	client *mysql.Client
}

func newBaseDataStore(conf *config.Config) (datastoreapi.DataStore, error) {
	client := mysql.NewClient(conf)

	return &BaseDataStore{
		client: client,
	}, nil
}

// PersistMetrics persists metrics to storage
func (store *BaseDataStore) PersistMetrics(condition *datastoreapi.Condition, jobMetrics *pb.JobMetrics, extra interface{}) error {
	storeJobMetrics := store.QueryJobMetrics(jobMetrics.JobMeta.Uuid)
	log.Infof("Report job metrics is %s", jobMetrics.JobMeta)
	if jobMetrics.JobMeta.Name != "" {
		storeJobMetrics.JobName = jobMetrics.JobMeta.Name
	}
	if storeJobMetrics.CreatedAt.IsZero() {
		storeJobMetrics.CreatedAt = time.Now()
	}

	switch jobMetrics.MetricsType {
	case pb.MetricsType_Job_Exit_Reason:
		return store.persistExitReason(storeJobMetrics, jobMetrics.GetJobExitReason())
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
		return store.persistModelFeature(storeJobMetrics, modelFeature)
	case pb.MetricsType_Runtime_Info:
		return store.persistRuntimeInfo(storeJobMetrics, jobMetrics.GetRuntimeInfo())
	case pb.MetricsType_Training_Hyper_Params:
		in := jobMetrics.GetTrainingHyperParams()
		trainingHyperParams := &common.TrainingHyperParams{
			BatchSize: uint64(in.GetBatchSize()),
			Epoch:     int(in.GetEpoch()),
			MaxSteps:  uint64(in.GetMaxSteps()),
		}
		return store.persistTrainingHyperParams(storeJobMetrics, trainingHyperParams)
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
		return store.persistWorkflowFeature(storeJobMetrics, workflowFeature)
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
		return store.persistTrainingSetFeature(storeJobMetrics, trainingSetFeature)
	case pb.MetricsType_Type:
		return store.persistType(storeJobMetrics, jobMetrics.GetType())
	case pb.MetricsType_Resource:
		return store.persistResource(storeJobMetrics, jobMetrics.GetResource())
	case pb.MetricsType_Customized_Data:
		return store.persistCustomizedData(storeJobMetrics, jobMetrics.GetCustomizedData())
	case pb.MetricsType_Optimization:
		return store.persistOptimization(storeJobMetrics, jobMetrics.GetJobOptimization())
	}
	return fmt.Errorf("[ElasticDataStore] invalid type: %s", condition.Type)
}

// GetData returns data for a given condition
func (store *BaseDataStore) GetData(condition *datastoreapi.Condition, data interface{}) error {
	return dsimplutils.GetData(store.client, condition, data)
}

func (store *BaseDataStore) persistTrainingHyperParams(jobMetrics *mysql.JobMetrics, params *common.TrainingHyperParams) error {
	hyperParams := &common.TrainingHyperParams{}
	err := json.Unmarshal([]byte(jobMetrics.HyperParamsFeature), hyperParams)
	if err != nil {
		hyperParams = params
	} else {
		// ElasticDL may update training hyper parameters during training
		hyperParams.Update(params)
	}
	jsonVal, _ := json.Marshal(hyperParams)
	jobMetrics.HyperParamsFeature = string(jsonVal)
	err = store.client.JobMetricsRecorder.Upsert(jobMetrics)
	return err
}

func (store *BaseDataStore) persistWorkflowFeature(jobMetrics *mysql.JobMetrics, params *common.WorkflowFeature) error {
	jsonVal, _ := json.Marshal(params)
	jobMetrics.JobFeature = string(jsonVal)
	err := store.client.JobMetricsRecorder.Upsert(jobMetrics)
	return err
}

func (store *BaseDataStore) persistTrainingSetFeature(jobMetrics *mysql.JobMetrics, params *common.TrainingSetFeature) error {
	trainingSet := &common.TrainingSetFeature{}
	err := json.Unmarshal([]byte(jobMetrics.DataSetFeature), trainingSet)
	if err != nil {
		trainingSet = params
	} else {
		// ElasticDL may update training dataset metrics during training
		trainingSet.Update(params)
	}
	jsonVal, _ := json.Marshal(trainingSet)
	jobMetrics.DataSetFeature = string(jsonVal)
	err = store.client.JobMetricsRecorder.Upsert(jobMetrics)
	return err
}

func (store *BaseDataStore) persistModelFeature(jobMetrics *mysql.JobMetrics, params *common.ModelFeature) error {
	jsonVal, _ := json.Marshal(params)
	jobMetrics.ModelFeature = string(jsonVal)
	err := store.client.JobMetricsRecorder.Upsert(jobMetrics)
	return err
}

func (store *BaseDataStore) persistRuntimeInfo(jobMetrics *mysql.JobMetrics, in *pb.RuntimeInfo) error {
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
	err = store.client.JobMetricsRecorder.Upsert(jobMetrics)
	return err
}

func (store *BaseDataStore) persistExitReason(jobMetrics *mysql.JobMetrics, reason string) error {
	jobMetrics.ExitReason = reason
	err := store.client.JobMetricsRecorder.Upsert(jobMetrics)
	return err
}

func (store *BaseDataStore) persistOptimization(jobMetrics *mysql.JobMetrics, in *pb.JobOptimization) error {
	var jobOptimizations []*pb.JobOptimization
	if jobMetrics.Optimization != "" {
		err := json.Unmarshal([]byte(jobMetrics.Optimization), &jobOptimizations)
		if err != nil {
			log.Errorf("Fail to unmarshal %s optimization json: %v", jobMetrics.JobUUID, err)
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
		log.Errorf("Fail to dump %s optimization json: %v", jobMetrics.JobUUID, err)
	}

	jobMetrics.Optimization = string(optimizationVal)
	err = store.client.JobMetricsRecorder.Upsert(jobMetrics)
	return err
}

func (store *BaseDataStore) persistType(jobMetrics *mysql.JobMetrics, tp string) error {
	jobMetrics.Type = tp
	err := store.client.JobMetricsRecorder.Upsert(jobMetrics)
	return err
}

func (store *BaseDataStore) persistResource(jobMetrics *mysql.JobMetrics, resource string) error {
	jobMetrics.Resource = resource
	err := store.client.JobMetricsRecorder.Upsert(jobMetrics)
	return err
}

func (store *BaseDataStore) persistCustomizedData(jobMetrics *mysql.JobMetrics, data string) error {
	jobMetrics.CustomizedData = data
	err := store.client.JobMetricsRecorder.Upsert(jobMetrics)
	return err
}

// QueryJobMetrics queries a JobMetrics by a job UUID
func (store *BaseDataStore) QueryJobMetrics(jobUUID string) *mysql.JobMetrics {
	cond := &mysql.JobMetricsCondition{
		JobUUID: jobUUID,
	}
	jobMetrics := &mysql.JobMetrics{}
	err := store.client.JobMetricsRecorder.Get(cond, jobMetrics)
	if err != nil {
		log.Infof("Not found job %s in db, insert a new one.", jobUUID)
		jobMetrics.JobUUID = jobUUID
	}
	return jobMetrics
}

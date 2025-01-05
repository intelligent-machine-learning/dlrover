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

package common

// TrainingHyperParams includes hyper parameters configured by users
type TrainingHyperParams struct {
	BatchSize uint64 `json:"batch_size"`
	Epoch     int    `json:"epoch"`
	MaxSteps  uint64 `json:"max_steps,omitempty"`
}

// WorkflowFeature describe the background of the job
// For example. The jobs in a workflow executed at different time
// probably have the same features.
type WorkflowFeature struct {
	UserID      string `json:"user_id"`
	JobName     string `json:"job_name"`
	CodeAddress string `json:"code_address"`
	WorkflowID  string `json:"workflow_id"`
	NodeID      string `json:"node_id,omitempty"`
	OdpsProject string `json:"odps_project,omitempty"`
	IsProd      bool   `json:"is_prod,omitempty"`
	Cluster     string `json:"cluster,omitempty"`
}

// TrainingSetFeature includes Training set features
type TrainingSetFeature struct {
	DatasetSize         uint64 `json:"dataset_size"`
	DatasetName         string `json:"dataset_name"`
	SparseItemCount     uint64 `json:"sparse_item_count,omitempty"`
	SparseFeatures      string `json:"sparse_features,omitempty"`
	SparseFeatureGroups string `json:"sparse_feature_groups,omitempty"`
	SparseFeatureShapes string `json:"sparse_feature_shapes,omitempty"`
	DenseFeatures       string `json:"dense_features,omitempty"`
	DenseFeatureShapes  string `json:"dense_feature_shapes,omitempty"`
	StorageSize         uint64 `json:"storage_size,omitempty"`
}

// ModelFeature includes model features
type ModelFeature struct {
	VariableCount      uint64           `json:"variable_count"`
	OpCount            uint64           `json:"op_count"`
	EmbeddingDimension uint64           `json:"embedding_dimension,omitempty"`
	TotalVariableSize  uint64           `json:"total_variable_size,omitempty"`
	MaxVariableSize    uint64           `json:"max_variable_size,omitempty"`
	UpdateOpCount      uint64           `json:"update_op_count,omitempty"`
	ReadOpCount        uint64           `json:"read_op_count,omitempty"`
	InputFetchDur      uint64           `json:"input_fetch_dur,omitempty"`
	Flops              uint64           `json:"flops,omitempty"`
	RecvOpCount        uint64           `json:"recv_op_count,omitempty"`
	KvEmbeddingDims    []int64          `json:"kv_embedding_dims,omitempty"`
	TensorAllocBytes   map[string]int64 `json:"tensor_alloc_bytes,omitempty"`
}

// JobRuntimeInfo includes job runtime information
type JobRuntimeInfo struct {
	GlobalStep   uint64             `json:"global_step"`
	TimeStamp    uint64             `json:"time_stamp"`
	Speed        float64            `json:"speed"`
	WorkerMemory map[uint64]float64 `json:"worker_memory"`
	WorkerCPU    map[uint64]float64 `json:"worker_cpu"`
	PSMemory     map[uint64]float64 `json:"ps_memory"`
	PSCPU        map[uint64]float64 `json:"ps_cpu"`
}

// Update updates TrainingHyperParams by a new TrainingHyperParams object.
func (hyperParams *TrainingHyperParams) Update(newHyperParams *TrainingHyperParams) {
	if newHyperParams.BatchSize > 0 {
		hyperParams.BatchSize = newHyperParams.BatchSize
	}
	if newHyperParams.Epoch > 0 {
		hyperParams.Epoch = newHyperParams.Epoch
	}
	if newHyperParams.MaxSteps > 0 {
		hyperParams.MaxSteps = newHyperParams.MaxSteps
	}
}

// Update updates TrainingSetFeature by a new TrainingSetFeature object.
func (trainingSet *TrainingSetFeature) Update(newTrainingSet *TrainingSetFeature) {
	if newTrainingSet.DatasetSize > 0 {
		trainingSet.DatasetSize = newTrainingSet.DatasetSize
	}
	if newTrainingSet.DatasetName != "" {
		trainingSet.DatasetName = newTrainingSet.DatasetName
	}
	if newTrainingSet.SparseItemCount > 0 {
		trainingSet.SparseItemCount = newTrainingSet.SparseItemCount
	}
	if newTrainingSet.SparseFeatures != "" {
		trainingSet.SparseFeatures = newTrainingSet.SparseFeatures
	}
	if newTrainingSet.SparseFeatureGroups != "" {
		trainingSet.SparseFeatureGroups = newTrainingSet.SparseFeatureGroups
	}
	if newTrainingSet.SparseFeatureShapes != "" {
		trainingSet.SparseFeatureShapes = newTrainingSet.SparseFeatureShapes
	}
	if newTrainingSet.DenseFeatures != "" {
		trainingSet.DenseFeatures = newTrainingSet.DenseFeatures
	}
	if newTrainingSet.DenseFeatureShapes != "" {
		trainingSet.DenseFeatureShapes = newTrainingSet.DenseFeatureShapes
	}
	if newTrainingSet.StorageSize > 0 {
		trainingSet.StorageSize = newTrainingSet.StorageSize
	}
}

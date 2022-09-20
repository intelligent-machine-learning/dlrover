// Copyright 2022 The EasyDL Authors. All rights reserved.
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

package controllers

import (
	elasticv1alpha1 "github.com/intelligent-machine-learning/easydl/operator/api/v1alpha1"
	common "github.com/intelligent-machine-learning/easydl/operator/pkg/common"
	commonv1 "github.com/intelligent-machine-learning/easydl/operator/pkg/common/api/v1"
)

const (
	// ReplicaTypeEasydlMaster is the type for easydl Master replica.
	ReplicaTypeEasydlMaster commonv1.ReplicaType = "easydl-master"

	// ReplicaTypeWorker is the type for training worker replica.
	ReplicaTypeWorker commonv1.ReplicaType = "worker"

	// ReplicaTypeParameterServer is the type for training parameter server replica
	ReplicaTypeParameterServer commonv1.ReplicaType = "ps"

	// ReplicaTypeEvaluator is the type for elaluator replica
	ReplicaTypeEvaluator commonv1.ReplicaType = "evaluator"
)

// initializeTrainingJobStatuses initializes the ReplicaStatuses for TrainingJob.
func initializeJobStatuses(jobStatus *elasticv1alpha1.ElasticJobStatus, rtype commonv1.ReplicaType) {
	initializeJobStatus(jobStatus)

	replicaType := commonv1.ReplicaType(rtype)
	jobStatus.ReplicaStatuses[replicaType] = &commonv1.ReplicaStatus{}
}

// initializeTrainingJobStatuses initializes the ReplicaStatuses for TrainingJob.
func initializeJobStatus(jobStatus *elasticv1alpha1.ElasticJobStatus) {
	if jobStatus.ReplicaStatuses == nil {
		jobStatus.ReplicaStatuses = make(map[commonv1.ReplicaType]*commonv1.ReplicaStatus)
	}
	if len(jobStatus.Conditions) == 0 {
		jobStatus.Conditions = []commonv1.JobCondition{}
	}
}

// updateJobConditions adds to the jobStatus a new condition if needed, with the conditionType, reason, and message.
func updateStatus(jobStatus *elasticv1alpha1.ElasticJobStatus, conditionType commonv1.JobConditionType, reason, message string) error {
	updateJobConditions(jobStatus, conditionType, reason, message)
	updatePhase(jobStatus, conditionType)
	return nil
}

// updateJobConditions adds to the jobStatus a new condition if needed, with the conditionType, reason, and message.
func updateJobConditions(status *elasticv1alpha1.ElasticJobStatus, conditionType commonv1.JobConditionType, reason, message string) error {
	condition := common.NewCondition(conditionType, reason, message)
	common.SetCondition(&status.JobStatus, condition)
	return nil
}

// isFailed checks if the job is failed.
func isFailed(status elasticv1alpha1.ElasticJobStatus) bool {
	return common.HasCondition(status.JobStatus, commonv1.JobFailed)
}

// isRunning checks if the job is running.
func isRunning(status elasticv1alpha1.ElasticJobStatus) bool {
	return common.HasCondition(status.JobStatus, commonv1.JobRunning)
}

func updatePhase(jobStatus *elasticv1alpha1.ElasticJobStatus, status commonv1.JobConditionType) {
	jobStatus.Phase = status
}

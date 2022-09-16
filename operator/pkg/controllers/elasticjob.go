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
	commonv1 "github.com/kubeflow/common/pkg/apis/common/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

const (
	// ReplicaTypeEasydlMaster is the type for easydl Master replica.
	ReplicaTypeEasydlMaster commonv1.ReplicaType = "easydl-master"

	// ReplicaTypeWorker is the type for training worker replica.
	ReplicaTypeWorker commonv1.ReplicaType = "worker"

	// ReplicaTypeParameterServer is the type for training parameter server replica
	ReplicaTypeParameterServer commonv1.ReplicaType = "parameter_server"

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
func updateJobConditions(jobStatus *elasticv1alpha1.ElasticJobStatus, conditionType commonv1.JobConditionType, reason, message string) error {
	condition := newCondition(conditionType, reason, message)
	setCondition(jobStatus, condition)
	return nil
}

// newCondition creates a new job condition.
func newCondition(conditionType commonv1.JobConditionType, reason, message string) commonv1.JobCondition {
	return commonv1.JobCondition{
		Type:               conditionType,
		Status:             corev1.ConditionTrue,
		LastUpdateTime:     metav1.Now(),
		LastTransitionTime: metav1.Now(),
		Reason:             reason,
		Message:            message,
	}
}

// setCondition updates the job to include the provided condition.
// If the condition that we are about to add already exists
// and has the same status and reason then we are not going to update.
func setCondition(status *elasticv1alpha1.ElasticJobStatus, condition commonv1.JobCondition) {
	// Do nothing if JobStatus have failed condition
	if isFailed(*status) {
		return
	}

	currentCond := getCondition(*status, condition.Type)

	// Do nothing if condition doesn't change
	if currentCond != nil && currentCond.Status == condition.Status &&
		currentCond.Reason == condition.Reason &&
		currentCond.Message == condition.Message {
		return
	}

	// Do not update lastTransitionTime if the status of the condition doesn't change.
	if currentCond != nil && currentCond.Status == condition.Status {
		condition.LastTransitionTime = currentCond.LastTransitionTime
	}

	// Append the updated condition to the conditions
	newConditions := filterOutCondition(status.Conditions, condition.Type)
	status.Conditions = append(newConditions, condition)
}

// getCondition returns the condition with the provided type.
func getCondition(status elasticv1alpha1.ElasticJobStatus, condType commonv1.JobConditionType) *commonv1.JobCondition {
	for _, condition := range status.Conditions {
		if condition.Type == condType {
			return &condition
		}
	}
	return nil
}

// filterOutCondition returns a new slice of Job conditions without conditions with the provided type.
func filterOutCondition(conditions []commonv1.JobCondition, condType commonv1.JobConditionType) []commonv1.JobCondition {
	var newConditions []commonv1.JobCondition
	for _, c := range conditions {
		if condType == commonv1.JobRestarting && c.Type == commonv1.JobRunning {
			continue
		}
		if condType == commonv1.JobRunning && c.Type == commonv1.JobRestarting {
			continue
		}

		if c.Type == condType {
			continue
		}

		// Set the running condition status to be false when current condition failed or succeeded
		if (condType == commonv1.JobFailed || condType == commonv1.JobSucceeded) && (c.Type == commonv1.JobRunning || c.Type == commonv1.JobFailed) {
			c.Status = corev1.ConditionFalse
		}

		newConditions = append(newConditions, c)
	}
	return newConditions
}

// isFailed checks if the job is failed.
func isFailed(status elasticv1alpha1.ElasticJobStatus) bool {
	return hasCondition(status, commonv1.JobFailed)
}

func hasCondition(status elasticv1alpha1.ElasticJobStatus, condType commonv1.JobConditionType) bool {
	for _, condition := range status.Conditions {
		if condition.Type == condType && condition.Status == corev1.ConditionTrue {
			return true
		}
	}
	return false
}

func updatePhase(jobStatus *elasticv1alpha1.ElasticJobStatus, status commonv1.JobConditionType) {
	jobStatus.Phase = status
}

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

package commmon

import (
	apiv1 "github.com/intelligent-machine-learning/easydl/operator/pkg/common/api/v1"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

const (
	// JobCreatedReason is added in a job when it is created.
	JobCreatedReason = "JobCreated"
	// JobSucceededReason is added in a job when it is succeeded.
	JobSucceededReason = "JobSucceeded"
	// JobRunningReason is added in a job when it is running.
	JobRunningReason = "JobRunning"
	// JobFailedReason is added in a job when it is failed.
	JobFailedReason = "JobFailed"
	// JobRestartingReason is added in a job when it is restarting.
	JobRestartingReason = "JobRestarting"
	// JobPendingReason is added in a job when it is pending.
	JobPendingReason = "JobPending"
	// JobScalingReason is added in a job when it is scaling up/down Pods.
	JobScalingReason = "JobScaling"

	// labels for pods and servers.

)

// IsSucceeded checks if the job is succeeded
func IsSucceeded(status apiv1.JobStatus) bool {
	return HasCondition(status, apiv1.JobSucceeded)
}

// IsFailed checks if the job is failed
func IsFailed(status apiv1.JobStatus) bool {
	return HasCondition(status, apiv1.JobFailed)
}

// UpdateJobConditions adds to the jobStatus a new condition if needed, with the conditionType, reason, and message
func UpdateJobConditions(jobStatus *apiv1.JobStatus, conditionType apiv1.JobConditionType, reason, message string) error {
	condition := NewCondition(conditionType, reason, message)
	SetCondition(jobStatus, condition)
	return nil
}

// HasCondition check wether there is the conditionType in jobStatus.
func HasCondition(status apiv1.JobStatus, condType apiv1.JobConditionType) bool {
	for _, condition := range status.Conditions {
		if condition.Type == condType && condition.Status == v1.ConditionTrue {
			return true
		}
	}
	return false
}

// NewCondition creates a new job condition.
func NewCondition(conditionType apiv1.JobConditionType, reason, message string) apiv1.JobCondition {
	return apiv1.JobCondition{
		Type:               conditionType,
		Status:             v1.ConditionTrue,
		LastUpdateTime:     metav1.Now(),
		LastTransitionTime: metav1.Now(),
		Reason:             reason,
		Message:            message,
	}
}

// GetCondition returns the condition with the provided type.
func GetCondition(status apiv1.JobStatus, condType apiv1.JobConditionType) *apiv1.JobCondition {
	for _, condition := range status.Conditions {
		if condition.Type == condType {
			return &condition
		}
	}
	return nil
}

// SetCondition updates the job to include the provided condition.
// If the condition that we are about to add already exists
// and has the same status and reason then we are not going to update.
func SetCondition(status *apiv1.JobStatus, condition apiv1.JobCondition) {
	// Do nothing if JobStatus have failed condition
	if IsFailed(*status) {
		return
	}

	currentCond := GetCondition(*status, condition.Type)

	// Do nothing if condition doesn't change
	if currentCond != nil && currentCond.Status == condition.Status && currentCond.Reason == condition.Reason {
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

// filterOutCondition returns a new slice of job conditions without conditions with the provided type.
func filterOutCondition(conditions []apiv1.JobCondition, condType apiv1.JobConditionType) []apiv1.JobCondition {
	var newConditions []apiv1.JobCondition
	for _, c := range conditions {
		if condType == apiv1.JobRestarting && c.Type == apiv1.JobRunning {
			continue
		}
		if condType == apiv1.JobRunning && c.Type == apiv1.JobRestarting {
			continue
		}

		if c.Type == condType {
			continue
		}

		// Set the running condition status to be false when current condition failed or succeeded
		if (condType == apiv1.JobFailed || condType == apiv1.JobSucceeded) && c.Type == apiv1.JobRunning {
			c.Status = v1.ConditionFalse
		}

		newConditions = append(newConditions, c)
	}
	return newConditions
}

// Copyright 2025 The DLRover Authors. All rights reserved.
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
	"testing"

	"github.com/intelligent-machine-learning/dlrover/go/elasticjob/api/v1alpha1"
	apiv1 "github.com/intelligent-machine-learning/dlrover/go/elasticjob/pkg/common/api/v1"
	"github.com/stretchr/testify/assert"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sigs.k8s.io/controller-runtime/pkg/event"
)

func TestProcessPodEvent(t *testing.T) {
	reconciler := &ElasticJobReconciler{}
	elasticJob := &v1alpha1.ElasticJob{
		ObjectMeta: metav1.ObjectMeta{
			Name:        "test",
			Namespace:   "dlrover",
			Annotations: map[string]string{},
			Labels:      map[string]string{},
		},
	}
	elasticJob.Status.ReplicaStatuses = make(map[apiv1.ReplicaType]*apiv1.ReplicaStatus)
	elasticJob.Status.ReplicaStatuses["worker"] = &apiv1.ReplicaStatus{}
	reconciler.CachedJobs = make(map[string]*v1alpha1.ElasticJob)
	reconciler.CachedJobs["test"] = elasticJob

	workerPod := &corev1.Pod{
		TypeMeta: metav1.TypeMeta{
			APIVersion: "v1",
			Kind:       "Pod",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name: "test-job-worker-0",
			OwnerReferences: []metav1.OwnerReference{
				*metav1.NewControllerRef(elasticJob, v1alpha1.SchemeGroupVersionKind),
			},
			Labels: map[string]string{"elasticjob.dlrover/replica-type": "worker"},
		},
		Spec: corev1.PodSpec{},
	}

	// Test a pod is created.
	workerPod.Status.Phase = corev1.PodPending
	createEvent := &event.CreateEvent{Object: workerPod}
	reconciler.ProcessPodCreateEvent(createEvent)
	assert.Equal(t, elasticJob.Status.ReplicaStatuses["worker"].Pending, int32(1))

	// Test a pending pod starts running.
	newPod := workerPod.DeepCopy()
	newPod.Status.Phase = corev1.PodRunning
	newPod.ResourceVersion = "1234"
	updateEvent := &event.UpdateEvent{
		ObjectNew: newPod,
		ObjectOld: workerPod,
	}
	reconciler.ProcessPodUpdateEvent(updateEvent)
	assert.Equal(t, elasticJob.Status.ReplicaStatuses["worker"].Pending, int32(0))
	assert.Equal(t, elasticJob.Status.ReplicaStatuses["worker"].Active, int32(1))

	// Test a running pod succeeded.
	workerPod.Status.Phase = corev1.PodRunning
	newPod.Status.Phase = corev1.PodSucceeded
	updateEvent = &event.UpdateEvent{
		ObjectNew: newPod,
		ObjectOld: workerPod,
	}
	reconciler.ProcessPodUpdateEvent(updateEvent)
	assert.Equal(t, elasticJob.Status.ReplicaStatuses["worker"].Active, int32(0))
	assert.Equal(t, elasticJob.Status.ReplicaStatuses["worker"].Succeeded, int32(1))

	// Test a running pod is deleted
	elasticJob.Status.ReplicaStatuses["worker"].Active = 1
	workerPod.Status.Phase = corev1.PodRunning
	deleteEvent := &event.DeleteEvent{Object: workerPod}
	reconciler.ProcessPodDeleteEvent(deleteEvent)
	assert.Equal(t, elasticJob.Status.ReplicaStatuses["worker"].Active, int32(0))
	assert.Equal(t, elasticJob.Status.ReplicaStatuses["worker"].Failed, int32(1))

	// Test a succeeded pod is deleted
	workerPod.Status.Phase = corev1.PodSucceeded
	deleteEvent = &event.DeleteEvent{Object: workerPod}
	reconciler.ProcessPodDeleteEvent(deleteEvent)
	assert.Equal(t, elasticJob.Status.ReplicaStatuses["worker"].Active, int32(0))
	assert.Equal(t, elasticJob.Status.ReplicaStatuses["worker"].Failed, int32(1))
}

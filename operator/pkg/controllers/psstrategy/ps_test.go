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

package psstrategy

import (
	elasticv1alpha1 "github.com/intelligent-machine-learning/easydl/operator/api/v1alpha1"
	commonv1 "github.com/intelligent-machine-learning/easydl/operator/pkg/common/api/v1"
	controllers "github.com/intelligent-machine-learning/easydl/operator/pkg/controllers"
	"github.com/stretchr/testify/assert"
	corev1 "k8s.io/api/core/v1"
	"testing"
)

func TestNewPSPod(t *testing.T) {
	job := newTestJob()

	container := corev1.Container{
		Name:            "main",
		Image:           "test",
		ImagePullPolicy: corev1.PullAlways,
		Command:         []string{"/bin/bash", "echo 0"},
	}

	job.Spec.ReplicaSpecs[ReplicaTypePS] = &elasticv1alpha1.ReplicaSpec{
		ReplicaSpec: commonv1.ReplicaSpec{
			Template: corev1.PodTemplateSpec{
				Spec: corev1.PodSpec{
					Containers:    []corev1.Container{container},
					RestartPolicy: corev1.RestartPolicyNever,
				},
			},
		},
		RestartCount: 3,
	}

	manager := newPSManager()
	pod := manager.newParameterServer(job, 0)
	assert.Equal(t, pod.Name, "test-psstrategy-ps-0")
	assert.Equal(t, pod.Labels[LabelRestartCount], "3")
	assert.Equal(
		t,
		pod.Labels[controllers.LabelReplicaTypeKey],
		string(ReplicaTypePS),
	)
}


func TestNewPSService(t *testing.T) {
	job := newTestJob()
	manager := newPSManager()
	service := manager.newPSService(job, 0)
	assert.Equal(
		t,
		service.Spec.Selector[controllers.LabelReplicaTypeKey],
		string(ReplicaTypePS),
	)
	assert.Equal(t, service.Spec.Selector[controllers.LabelReplicaIndexKey], "0")
}
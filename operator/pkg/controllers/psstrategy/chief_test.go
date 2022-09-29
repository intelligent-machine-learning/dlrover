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
	"encoding/json"
	elasticv1alpha1 "github.com/intelligent-machine-learning/easydl/operator/api/v1alpha1"
	commonv1 "github.com/intelligent-machine-learning/easydl/operator/pkg/common/api/v1"
	controllers "github.com/intelligent-machine-learning/easydl/operator/pkg/controllers"
	"github.com/stretchr/testify/assert"
	corev1 "k8s.io/api/core/v1"
	"testing"
)

func TestNewChiefPod(t *testing.T) {
	job := newTestJob()
	container := corev1.Container{
		Name:            "main",
		Image:           "test",
		ImagePullPolicy: corev1.PullAlways,
		Command:         []string{"/bin/bash", "echo 0"},
	}

	job.Spec.ReplicaSpecs[ReplicaTypeChief] = &elasticv1alpha1.ReplicaSpec{
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

	manager := newChiefManager()
	pod := manager.newTask(job, 0)
	assert.Equal(t, pod.Name, "test-psstrategy-chief-0")
	assert.Equal(t, pod.Labels[LabelRestartCount], "3")
	assert.Equal(
		t,
		pod.Labels[controllers.LabelReplicaTypeKey],
		string(ReplicaTypeChief),
	)

	cluster := newTFcluster()
	manager.insertTfConfigToEnv(&container, cluster, 0)
	tfConfig := TFConfig{}
	err := json.Unmarshal([]byte(container.Env[0].Value), &tfConfig)
	assert.NoError(t, err)
	assert.Equal(t, tfConfig.Task.Type, ReplicaTypeChief)
	assert.Equal(t, tfConfig.Task.Index, 0)
}

func TestNewChiefService(t *testing.T) {
	job := newTestJob()
	manager := newChiefManager()
	service := manager.newTaskService(job, 0, chiefServicePort)
	assert.Equal(
		t,
		service.Spec.Selector[controllers.LabelReplicaTypeKey],
		string(ReplicaTypeChief),
	)
	assert.Equal(t, service.Spec.Selector[controllers.LabelReplicaIndexKey], "0")
}

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

package master

import (
	elasticv1alpha1 "github.com/intelligent-machine-learning/easydl/dlrover/go/operator/api/v1alpha1"
	commonv1 "github.com/intelligent-machine-learning/easydl/dlrover/go/operator/pkg/common/api/v1"
	"github.com/stretchr/testify/assert"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"strings"
	"testing"
)

func TestCreateMasterPod(t *testing.T) {
	job := &elasticv1alpha1.ElasticJob{
		ObjectMeta: metav1.ObjectMeta{
			Name:        "test-ps",
			Namespace:   "dlrover",
			Annotations: map[string]string{},
			Labels:      map[string]string{},
		},
	}
	job.Spec.ReplicaSpecs = make(map[commonv1.ReplicaType]*elasticv1alpha1.ReplicaSpec)
	NewMasterTemplateToJob(job, "dlrover-master:test")
	manager := &Manager{}
	pod := manager.newJobMaster(job, initMasterIndex)
	assert.Equal(t, pod.Name, "elasticjob-test-ps-dlrover-master")
	assert.Equal(t, pod.Spec.Containers[0].Image, "dlrover-master:test")
	assert.Equal(t, string(pod.Spec.Containers[0].ImagePullPolicy), "Always")
	assert.True(t, strings.Contains(pod.Spec.Containers[0].Command[2], "--namespace dlrover"))
	assert.True(t, strings.Contains(pod.Spec.Containers[0].Command[2], "--job_name test-ps"))
	assert.True(t, strings.Contains(pod.Spec.Containers[0].Command[2], "--port 50001"))
}

func TestCreateMasterPodWithImage(t *testing.T) {
	job := &elasticv1alpha1.ElasticJob{
		ObjectMeta: metav1.ObjectMeta{
			Name:        "test-ps",
			Namespace:   "dlrover",
			Annotations: map[string]string{},
			Labels:      map[string]string{},
		},
	}

	container := corev1.Container{
		Image:           "dlrover-master:test-v0",
		ImagePullPolicy: "Always",
	}
	job.Spec.ReplicaSpecs = make(map[commonv1.ReplicaType]*elasticv1alpha1.ReplicaSpec)
	job.Spec.ReplicaSpecs[ReplicaTypeJobMaster] = &elasticv1alpha1.ReplicaSpec{
		ReplicaSpec: commonv1.ReplicaSpec{
			Template: corev1.PodTemplateSpec{
				Spec: corev1.PodSpec{
					Containers:    []corev1.Container{container},
					RestartPolicy: corev1.RestartPolicyNever,
				},
			},
		},
	}

	NewMasterTemplateToJob(job, "dlrover-master:test")
	manager := &Manager{}
	pod := manager.newJobMaster(job, initMasterIndex)
	assert.Equal(t, pod.Name, "elasticjob-test-ps-dlrover-master")
	assert.Equal(t, pod.Spec.Containers[0].Image, "dlrover-master:test-v0")
	assert.Equal(t, string(pod.Spec.Containers[0].ImagePullPolicy), "Always")
}

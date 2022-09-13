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
	"github.com/stretchr/testify/assert"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"testing"
)

func TestCreatePod(t *testing.T) {
	job := &elasticv1alpha1.ElasticJob{
		ObjectMeta: metav1.ObjectMeta{
			Name:        "test-ps",
			Namespace:   "easydl",
			Annotations: map[string]string{},
			Labels:      map[string]string{},
		},
	}
	container := corev1.Container{
		Name:            "main",
		Image:           "test",
		ImagePullPolicy: corev1.PullAlways,
		Command:         []string{"python", "--version"},
	}
	podTemplate := &corev1.PodTemplateSpec{
		Spec: corev1.PodSpec{
			Containers: []corev1.Container{container},
		},
	}

	manager := newPodManager()
	pod := manager.CreatePod(job, podTemplate, "worker", 0)
	assert.Equal(t, pod.Labels[labelReplicaTypeKey], "worker")
	assert.Equal(t, pod.Name, "test-ps-worker-0")
	assert.Equal(t, pod.Spec.Containers[0].Image, "test")
}

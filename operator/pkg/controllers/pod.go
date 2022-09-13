/*
Copyright 2022.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package controllers

import (
	"fmt"
	"github.com/golang/glog"
	elasticv1alpha1 "github.com/intelligent-machine-learning/easydl/operator/api/v1alpha1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

const (
	labelAppName         = "app"
	labelReplicaTypeKey  = "replica-type"
	labelReplicaIndexKey = "replica-index"
	easydlApp            = "easydl"
)

// TaskType defines the task type (ps/worker) of a pod.
type TaskType string

// PodManager manages the lifecycle of a pod including creation, updation and deletion.
type PodManager struct{}

func newPodManager() *PodManager {
	return &PodManager{}
}

// CreatePod creates a Pod according to a PodTemplateSpec
func (m *PodManager) CreatePod(job *elasticv1alpha1.ElasticJob, podTemplate *corev1.PodTemplateSpec, taskType TaskType, taskIndex int) *corev1.Pod {
	podName := fmt.Sprintf("%s-%s-%d", job.GetName(), taskType, taskIndex)

	podSpec := podTemplate.DeepCopy()

	if len(podSpec.Labels) == 0 {
		podSpec.Labels = make(map[string]string)
	}

	if len(podSpec.Annotations) == 0 {
		podSpec.Annotations = make(map[string]string)
	}
	podSpec.Labels[labelAppName] = easydlApp
	podSpec.Labels[labelReplicaTypeKey] = string(taskType)
	podSpec.Labels[labelReplicaIndexKey] = fmt.Sprintf("%d", taskIndex)

	for key, value := range job.Labels {
		podSpec.Labels[key] = value
	}

	for key, value := range job.Annotations {
		podSpec.Annotations[key] = value
	}

	if len(podSpec.Spec.Containers) == 0 {
		glog.Errorf("Pod %s-%d does not have any container", taskType, taskIndex)
		return nil
	}

	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:        podName,
			Namespace:   job.Namespace,
			Labels:      podSpec.Labels,
			Annotations: podSpec.Annotations,
			OwnerReferences: []metav1.OwnerReference{
				*metav1.NewControllerRef(job, elasticv1alpha1.SchemeGroupVersionKind),
			},
		},
		Spec: podSpec.Spec,
	}
	return pod
}

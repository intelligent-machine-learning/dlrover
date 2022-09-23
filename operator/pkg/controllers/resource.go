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
	"context"
	"github.com/golang/glog"
	elasticv1alpha1 "github.com/intelligent-machine-learning/easydl/operator/api/v1alpha1"
	commonv1 "github.com/intelligent-machine-learning/easydl/operator/pkg/common/api/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

const (
	labelAppName = "app"
	labelJobName = "elasticjob-name"
	easydlApp    = "easydl"
)

// PodManager manages the lifecycle of a pod including creation, updation and deletion.
type PodManager struct{}

func newPodManager() *PodManager {
	return &PodManager{}
}

// NewPod creates a Pod according to a PodTemplateSpec
func (m *PodManager) NewPod(job *elasticv1alpha1.ElasticJob, podTemplate *corev1.PodTemplateSpec, podName string) *corev1.Pod {
	podSpec := podTemplate.DeepCopy()

	if len(podSpec.Labels) == 0 {
		podSpec.Labels = make(map[string]string)
	}

	if len(podSpec.Annotations) == 0 {
		podSpec.Annotations = make(map[string]string)
	}
	podSpec.Labels[labelAppName] = easydlApp
	podSpec.Labels[labelJobName] = job.Name

	for key, value := range job.Labels {
		podSpec.Labels[key] = value
	}

	for key, value := range job.Annotations {
		podSpec.Annotations[key] = value
	}

	if len(podSpec.Spec.Containers) == 0 {
		glog.Errorf("Pod %s does not have any container", podName)
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

// GetReplicaTypePods get all ReplicaType Pods of a job
func (m *PodManager) GetReplicaTypePods(
	r *ElasticJobReconciler,
	job *elasticv1alpha1.ElasticJob,
	replicaType commonv1.ReplicaType,
) ([]corev1.Pod, error) {
	replicaLabels := make(map[string]string)
	replicaLabels[LabelReplicaTypeKey] = string(replicaType)
	replicaLabels[labelJobName] = job.Name
	labelSelector := &metav1.LabelSelector{
		MatchLabels: replicaLabels,
	}
	selector, err := metav1.LabelSelectorAsSelector(labelSelector)
	podlist := &corev1.PodList{}
	err = r.List(context.Background(), podlist, client.MatchingLabelsSelector{Selector: selector})
	if err != nil {
		return nil, err
	}
	return podlist.Items, nil
}

// GetReplicaStatus gets ReplicaStatus from ReplicaType Pods
func (m *PodManager) GetReplicaStatus(pods []corev1.Pod) *commonv1.ReplicaStatus {
	replicaStatus := commonv1.ReplicaStatus{}

	for _, pod := range pods {
		if pod.Status.Phase == corev1.PodPending {
			replicaStatus.Pending++
		} else if pod.Status.Phase == corev1.PodRunning {
			replicaStatus.Active++
		} else if pod.Status.Phase == corev1.PodFailed {
			replicaStatus.Failed++
		} else if pod.Status.Phase == corev1.PodSucceeded {
			replicaStatus.Succeeded++
		}
	}
	return &replicaStatus
}

// NewService create a service
func (m *PodManager)NewService(job *elasticv1alpha1.ElasticJob, name string, port int32, selector map[string]string) *corev1.Service {
	selector[labelAppName] = easydlApp
	selector[labelJobName] = job.Name
	return &corev1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: job.Namespace,
			Labels:    selector,
			OwnerReferences: []metav1.OwnerReference{
				*metav1.NewControllerRef(job, elasticv1alpha1.SchemeGroupVersionKind),
			},
		},
		Spec: corev1.ServiceSpec{
			ClusterIP: "None",
			Selector:  selector,
			Ports: []corev1.ServicePort{
				{
					Port: port,
				},
			},
		},
	}
}
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

package common

import (
	"context"
	"github.com/golang/glog"
	elasticv1alpha1 "github.com/intelligent-machine-learning/easydl/dlrover/go/operator/api/v1alpha1"
	commonv1 "github.com/intelligent-machine-learning/easydl/dlrover/go/operator/pkg/common/api/v1"
	logger "github.com/sirupsen/logrus"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	utilpointer "k8s.io/utils/pointer"
	runtime_client "sigs.k8s.io/controller-runtime/pkg/client"
)

const (
	labelAppName = "app"
	labelJobName = "elasticjob-name"
	appName      = "dlrover"
	// LabelReplicaTypeKey is the key of ReplicaType in labels
	LabelReplicaTypeKey = "elasticjob.dlrover/replica-type"
	// LabelReplicaIndexKey is the key of ReplicaIndex in labels
	LabelReplicaIndexKey = "elasticjob.dlrover/replica-index"
	// LabelRankIndexKey is the key of rankIndex.
	LabelRankIndexKey = "elasticjob.dlrover/rank-index"
)

// ReplicaManagers contains the manager for each ReplicaType
var ReplicaManagers = make(map[commonv1.ReplicaType]ReplicaManager)

// ReplicaManager manage pods of ReplicaType
type ReplicaManager interface {
	ReconcilePods(client runtime_client.Client, job *elasticv1alpha1.ElasticJob, scalePlan *elasticv1alpha1.ScalePlan) error

	SyncJobState(client runtime_client.Client, job *elasticv1alpha1.ElasticJob) error

	HandleFaultPods(client runtime_client.Client, job *elasticv1alpha1.ElasticJob) error

	StopRunningPods(client runtime_client.Client, job *elasticv1alpha1.ElasticJob) error
}

// NewPod creates a Pod according to a PodTemplateSpec
func NewPod(
	job *elasticv1alpha1.ElasticJob,
	podTemplate *corev1.PodTemplateSpec,
	podName string,
) *corev1.Pod {
	podSpec := podTemplate.DeepCopy()
	if len(podSpec.Labels) == 0 {
		podSpec.Labels = make(map[string]string)
	}

	if len(podSpec.Annotations) == 0 {
		podSpec.Annotations = make(map[string]string)
	}
	podSpec.Labels[labelAppName] = appName
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

// DeletePod remove a Pod
func DeletePod(client runtime_client.Client, pod *corev1.Pod) error {
	deleteOptions := &runtime_client.DeleteOptions{GracePeriodSeconds: utilpointer.Int64Ptr(0)}
	err := client.Delete(context.Background(), pod, deleteOptions)
	return err
}

// GetReplicaTypePods get all ReplicaType Pods of a job
func GetReplicaTypePods(
	client runtime_client.Client,
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
	if err != nil {
		logger.Warningf("No selector found")
	}
	podlist := &corev1.PodList{}
	err = client.List(
		context.Background(),
		podlist,
		runtime_client.MatchingLabelsSelector{Selector: selector},
	)
	if err != nil {
		return nil, err
	}
	return podlist.Items, nil
}

// GetPod gets a Pod object
func GetPod(
	client runtime_client.Client,
	namespace string,
	name string,
) (*corev1.Pod, error) {
	pod := &corev1.Pod{}
	err := client.Get(context.TODO(), types.NamespacedName{
		Name:      name,
		Namespace: namespace,
	}, pod)
	if err != nil {
		logger.Warningf("Failed to get Pod : %s, error: %v", name, err)
		return nil, err
	}
	return pod, nil
}

// GetReplicaStatus gets ReplicaStatus from ReplicaType Pods
func GetReplicaStatus(pods []corev1.Pod) *commonv1.ReplicaStatus {
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
func NewService(job *elasticv1alpha1.ElasticJob, name string, port int, selector map[string]string) *corev1.Service {
	selector[labelAppName] = appName
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
			Selector: selector,
			Type:     corev1.ServiceTypeClusterIP,
			Ports: []corev1.ServicePort{
				{
					Port: int32(port),
				},
			},
		},
	}
}

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
	"context"
	"fmt"
	elasticv1alpha1 "github.com/intelligent-machine-learning/easydl/operator/api/v1alpha1"
	controllers "github.com/intelligent-machine-learning/easydl/operator/pkg/controllers"
	logger "github.com/sirupsen/logrus"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
)

const (
	psServicePort int32 = 3333
)

// PSManager generates a master pod object.
type PSManager struct {
	controllers.PodManager
}

func init() {
	controllers.ReplicaManagers[ReplicaTypePS] = newPSManager()
}

func newPSManager() *PSManager {
	return &PSManager{}
}

func insertCommonPSLabels(labels map[string]string, psIndex int32){
	labels[controllers.LabelReplicaTypeKey] = string(ReplicaTypePS)
	labels[controllers.LabelReplicaIndexKey] = fmt.Sprintf("%d", psIndex)
}

func (m *PSManager) newParameterServer(job *elasticv1alpha1.ElasticJob, psIndex int32) *corev1.Pod {
	spec, ok := job.Spec.ReplicaSpecs[ReplicaTypePS]
	if !ok {
		return nil
	}
	podName := m.newPSName(job.Name, psIndex)
	pod := m.NewPod(job, &spec.Template, podName)
	pod.Labels[LabelRestartCount] = fmt.Sprintf("%d", spec.RestartCount)
	insertCommonPSLabels(pod.Labels, psIndex)
	return pod
}

// ReconcilePods creates a Pod on a K8s cluster
func (m *PSManager) ReconcilePods(
	r *controllers.ElasticJobReconciler,
	job *elasticv1alpha1.ElasticJob,
	resourceSpec *elasticv1alpha1.ReplicaResourceSpec,
) error {
	for i := 0; i < resourceSpec.Replicas; i++ {
		psIndex := int32(i)
		ps := m.newParameterServer(job, psIndex)
		err := r.Create(context.Background(), ps)
		if err != nil {
			r.Recorder.Eventf(job, corev1.EventTypeWarning, string(corev1.PodFailed), "PS pod %s created failed: %v", ps.Name, err)
			return err
		}
		service := m.newPSService(job, psIndex)
		err = r.Create(context.Background(), service)
		if err != nil {
			r.Recorder.Eventf(job, corev1.EventTypeWarning, string(corev1.PodFailed), "worker service %s created failed: %v", service.Name, err)
			return err
		}
	}
	return nil
}

// SyncJobState synchronize the job status by replicas
func (m *PSManager) SyncJobState(
	r *controllers.ElasticJobReconciler,
	job *elasticv1alpha1.ElasticJob,
) error {
	psPods, err := m.GetReplicaTypePods(r, job, ReplicaTypePS)
	if errors.IsNotFound(err) {
		logger.Warningf("No any PS found: %v", err)
		return nil
	}
	psStatus := m.GetReplicaStatus(psPods)
	job.Status.ReplicaStatuses[ReplicaTypePS] = psStatus
	return nil
}

func (m *PSManager) newPSName(jobName string, psIndex int32) string {
	return fmt.Sprintf("%s-%s-%d", jobName, string(ReplicaTypePS), psIndex)
}

func (m *PSManager) newPSService(job *elasticv1alpha1.ElasticJob, psIndex int32) *corev1.Service {
	name := m.newPSName(job.Name, psIndex)
	selector := make(map[string]string)
	insertCommonPSLabels(selector, psIndex)
	service := m.NewService(job, name, psServicePort, selector)
	return service
}

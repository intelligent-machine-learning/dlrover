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
	"context"
	"fmt"
	elasticv1alpha1 "github.com/intelligent-machine-learning/easydl/operator/api/v1alpha1"
	common "github.com/intelligent-machine-learning/easydl/operator/pkg/common"
	commonv1 "github.com/intelligent-machine-learning/easydl/operator/pkg/common/api/v1"
	logger "github.com/sirupsen/logrus"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

const (
	initMasterContainerCPU     = "2"
	initMasterContainerMemory  = "4Gi"
	initMasterContainerStorage = "4Gi"
	masterCommand              = "sleep 30"
	masterImage                = "python:3.6.5"
	masterServicePort          = 50001

	// ReplicaTypeEasydlMaster is the type for easydl Master replica.
	ReplicaTypeEasydlMaster commonv1.ReplicaType = "easydl-master"
)

// MasterManager generates a master pod object.
type MasterManager struct {
	PodManager
}

func init() {
	ReplicaManagers[ReplicaTypeEasydlMaster] = newMasterManager()
}

func newMasterManager() *MasterManager {
	return &MasterManager{}
}

func (m *MasterManager) newEasydlMaster(job *elasticv1alpha1.ElasticJob) *corev1.Pod {
	container := corev1.Container{
		Name:            "main",
		Image:           masterImage,
		ImagePullPolicy: corev1.PullAlways,
		Command:         []string{"/bin/bash", "-c", masterCommand},
		Resources: corev1.ResourceRequirements{
			Requests: corev1.ResourceList{
				corev1.ResourceCPU:              resource.MustParse(initMasterContainerCPU),
				corev1.ResourceMemory:           resource.MustParse(initMasterContainerMemory),
				corev1.ResourceEphemeralStorage: resource.MustParse(initMasterContainerStorage),
			},
			Limits: corev1.ResourceList{
				corev1.ResourceCPU:              resource.MustParse(initMasterContainerCPU),
				corev1.ResourceMemory:           resource.MustParse(initMasterContainerMemory),
				corev1.ResourceEphemeralStorage: resource.MustParse(initMasterContainerStorage),
			},
		},
	}
	podTemplate := &corev1.PodTemplateSpec{
		Spec: corev1.PodSpec{
			Containers:    []corev1.Container{container},
			RestartPolicy: corev1.RestartPolicyNever,
		},
	}
	masterName := NewEasydlMasterName(job.Name)
	pod := m.NewPod(job, podTemplate, masterName)
	pod.Labels[LabelReplicaTypeKey] = string(ReplicaTypeEasydlMaster)
	return pod
}

// ReconcilePods reconciles Pods of a job on a K8s cluster
func (m *MasterManager) ReconcilePods(
	r *ElasticJobReconciler,
	job *elasticv1alpha1.ElasticJob,
	resourceSpec *elasticv1alpha1.ReplicaResourceSpec,
) error {
	masterPod := m.newEasydlMaster(job)
	err := r.Create(context.Background(), masterPod)
	if err != nil {
		r.Recorder.Eventf(job, corev1.EventTypeWarning, string(commonv1.JobFailed), "master pod created failed: %v", err)
		return err
	}
	service := m.newEasydlMasterService(job)
	err = r.Create(context.Background(), service)
	if err != nil {
		r.Recorder.Eventf(job, corev1.EventTypeWarning, string(corev1.PodFailed), "master service %s created failed: %v", service.Name, err)
		return err
	}
	return nil
}

// SyncJobState synchronize the job status by replicas
func (m *MasterManager) SyncJobState(r *ElasticJobReconciler, job *elasticv1alpha1.ElasticJob) error {
	master, err := m.getMasterPod(r, job)
	if master == nil {
		logger.Warnf("Failed to get master, error : %v", err)
		return nil
	}

	job.Status.ReplicaStatuses[ReplicaTypeEasydlMaster] = m.GetReplicaStatus([]corev1.Pod{*master})
	if master.Status.Phase == corev1.PodSucceeded {
		msg := fmt.Sprintf("job(%s/%s) successfully completed", job.Namespace, job.Name)
		r.Recorder.Event(job, corev1.EventTypeNormal, string(commonv1.JobSucceeded), msg)
		if job.Status.CompletionTime == nil {
			now := metav1.Now()
			job.Status.CompletionTime = &now
		}
		UpdateStatus(&job.Status, commonv1.JobSucceeded, common.JobCreatedReason, msg)
	} else if master.Status.Phase == corev1.PodFailed {
		msg := fmt.Sprintf("job(%s/%s) has failed", job.Namespace, job.Name)
		reason := master.Status.Reason
		if reason == "" {
			reason = common.JobFailedReason
		}
		r.Recorder.Event(job, corev1.EventTypeWarning, reason, msg)
		if job.Status.CompletionTime == nil {
			now := metav1.Now()
			job.Status.CompletionTime = &now
		}
		UpdateStatus(&job.Status, commonv1.JobFailed, reason, msg)
	} else if master.Status.Phase == corev1.PodPending {
		msg := fmt.Sprintf("job(%s/%s) is pending.", job.Namespace, job.Name)
		UpdateStatus(&job.Status, commonv1.JobPending, common.JobPendingReason, msg)
	} else if master.Status.Phase == corev1.PodRunning {
		if !isRunning(job.Status) {
			msg := fmt.Sprintf("job(%s/%s) is running.", job.Namespace, job.Name)
			UpdateStatus(&job.Status, commonv1.JobRunning, common.JobRunningReason, msg)
			r.Recorder.Event(job, corev1.EventTypeNormal, common.JobRunningReason, msg)
		}
	}
	return nil
}

// getMasterPod gets the master pod of a job from a cluster.
func (m *MasterManager) getMasterPod(r *ElasticJobReconciler, job *elasticv1alpha1.ElasticJob) (*corev1.Pod, error) {
	pods, err := m.GetReplicaTypePods(r.Client, job, ReplicaTypeEasydlMaster)
	if errors.IsNotFound(err) {
		return nil, err
	}
	if len(pods) == 0 {
		logger.Warnf("No master of job %s Found", job.Name)
		return nil, nil
	}
	return &pods[0], nil
}

func (m *MasterManager) newEasydlMasterService(job *elasticv1alpha1.ElasticJob) *corev1.Service {
	name := NewEasydlMasterName(job.Name)
	selector := make(map[string]string)
	selector[LabelReplicaTypeKey] = string(ReplicaTypeEasydlMaster)
	service := m.NewService(job, name, masterServicePort, selector)
	return service
}

// NewEasydlMasterName create a service name for Job master
func NewEasydlMasterName(jobName string) string {
	return fmt.Sprintf("%s-%s", jobName, string(ReplicaTypeEasydlMaster))
}

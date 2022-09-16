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
	elasticv1alpha1 "github.com/intelligent-machine-learning/easydl/operator/api/v1alpha1"
	commonv1 "github.com/kubeflow/common/pkg/apis/common/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
)

const (
	initMasterContainerCPU     = "2"
	initMasterContainerMemory  = "4Gi"
	initMasterContainerStorage = "4Gi"
	masterCommand              = "python -m elasticdl.python.master.main"
	masterImage                = "easydl/easydl-master:v1.0.0"
)

// MasterManager generates a master pod object.
type MasterManager struct {
	PodManager
}

func newMasterManager() *MasterManager {
	return &MasterManager{}
}

func (m *MasterManager) generateEasydlMaster(job *elasticv1alpha1.ElasticJob) *corev1.Pod {
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
			Containers: []corev1.Container{container},
		},
	}
	pod := m.GeneratePod(job, podTemplate, "easydl-master", 0)
	return pod
}

func (m *MasterManager) createPod(r *ElasticJobReconciler, job *elasticv1alpha1.ElasticJob) error {
	masterPod := m.generateEasydlMaster(job)
	err := r.Create(context.Background(), masterPod)
	if err != nil {
		r.Recorder.Eventf(job, corev1.EventTypeWarning, string(commonv1.JobFailed), "master pod created failed: %v", err)
		return err
	}
	return nil
}

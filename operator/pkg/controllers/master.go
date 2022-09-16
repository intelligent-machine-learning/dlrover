package controllers

import (
	elasticv1alpha1 "github.com/intelligent-machine-learning/easydl/operator/api/v1alpha1"
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

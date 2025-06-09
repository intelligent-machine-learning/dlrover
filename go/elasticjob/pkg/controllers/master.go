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

package controllers

import (
	"context"
	"fmt"

	elasticv1alpha1 "github.com/intelligent-machine-learning/dlrover/go/elasticjob/api/v1alpha1"
	common "github.com/intelligent-machine-learning/dlrover/go/elasticjob/pkg/common"
	commonv1 "github.com/intelligent-machine-learning/dlrover/go/elasticjob/pkg/common/api/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	runtime_client "sigs.k8s.io/controller-runtime/pkg/client"
)

const (
	initMasterContainerCPU     = "1"
	initMasterContainerMemory  = "2Gi"
	initMasterContainerStorage = "2Gi"
	masterCommand              = "python -m dlrover.python.master.main"
	masterServicePort          = 50001
	defaultImagePullPolicy     = "Always"
	envMasterAddrKey           = "DLROVER_MASTER_ADDR"
	envBrainServiceAddrKey     = "DLROVER_BRAIN_SERVICE_ADDR"
	defaultBrainServiceAddr    = "dlrover-brain.dlrover.svc.cluster.local:50001"
	envPodIP                   = "POD_IP"

	// JobMasterReplicaType is the type for DLRover ElasticJob Master replica.
	JobMasterReplicaType commonv1.ReplicaType = "dlrover-master"

	// supported arguments(should be supported in 'dlrover.python.master.args')
	pendingTimeoutArg      = "pending_timeout"
	pendingFailStrategyArg = "pending_fail_strategy"
	serviceType            = "service_type"
	preCheckOperatorsArg   = "pre_check_ops"
	preCheckBypassArg      = "pre_check_bypass"
)

// ReconcileJobMasterPod reconciles the job master Pods.
func ReconcileJobMasterPod(
	client runtime_client.Client,
	job *elasticv1alpha1.ElasticJob,
	masterIndex int32,
) error {
	masterPod := newElasticJobMaster(job, masterIndex)
	err := client.Create(context.Background(), masterPod)
	if err != nil && !errors.IsAlreadyExists(err) {
		return err
	}
	masterServiceName := newJobMasterServiceName(job.Name)
	service := &corev1.Service{}
	service = newJobMasterService(job, masterServiceName)
	err = client.Create(context.Background(), service)
	if errors.IsAlreadyExists(err) {
		return nil
	}
	return nil
}

func newElasticJobMaster(
	job *elasticv1alpha1.ElasticJob, replicaIndex int32,
) *corev1.Pod {
	masterName := newJobMasterName(job.Name, replicaIndex)
	pod := common.NewPod(
		job, &job.Spec.ReplicaSpecs[JobMasterReplicaType].Template, masterName,
	)
	pod.Labels[common.LabelReplicaTypeKey] = string(JobMasterReplicaType)
	pod.Labels[common.LabelReplicaIndexKey] = fmt.Sprintf("%d", replicaIndex)
	return pod
}

func newJobMasterService(job *elasticv1alpha1.ElasticJob, serviceName string) *corev1.Service {
	selector := make(map[string]string)
	selector[common.LabelReplicaTypeKey] = string(JobMasterReplicaType)
	service := common.NewService(job, serviceName, masterServicePort, selector)
	return service
}

// newJobMasterServiceName create a service name for Job master
func newJobMasterServiceName(jobName string) string {
	return fmt.Sprintf("elasticjob-%s-%s", jobName, string(JobMasterReplicaType))
}

// newJobMasterName create a name for Job master
func newJobMasterName(jobName string, replicaIndex int32) string {
	return fmt.Sprintf("elasticjob-%s-%s-%d", jobName, string(JobMasterReplicaType), replicaIndex)
}

func getMasterArguments() []string {
	return []string{pendingTimeoutArg, pendingFailStrategyArg, serviceType, preCheckOperatorsArg, preCheckBypassArg}
}

// SetDefaultMasterTemplateToJob sets configurations to the master template of a job.
func SetDefaultMasterTemplateToJob(job *elasticv1alpha1.ElasticJob, masterImage string) {
	var podTemplate *corev1.PodTemplateSpec
	if _, ok := job.Spec.ReplicaSpecs[JobMasterReplicaType]; ok {
		podTemplate = &job.Spec.ReplicaSpecs[JobMasterReplicaType].Template
	} else {
		podTemplate = createDefaultMasterTemplate(job, masterImage)
	}
	podIPEnv := corev1.EnvVar{
		Name: envPodIP,
		ValueFrom: &corev1.EnvVarSource{
			FieldRef: &corev1.ObjectFieldSelector{
				APIVersion: "v1",
				FieldPath:  "status.podIP",
			},
		},
	}
	podTemplate.Spec.Containers[0].Env = append(podTemplate.Spec.Containers[0].Env, podIPEnv)
	job.Spec.ReplicaSpecs[JobMasterReplicaType] = &elasticv1alpha1.ReplicaSpec{
		ReplicaSpec: commonv1.ReplicaSpec{
			Template: *podTemplate,
		},
	}
}

func createDefaultMasterTemplate(job *elasticv1alpha1.ElasticJob, masterImage string) *corev1.PodTemplateSpec {
	command := masterCommand + fmt.Sprintf(
		" --platform pyk8s --namespace %s --job_name %s --port %d",
		job.Namespace, job.Name, masterServicePort,
	)

	// for extra arguments
	for _, item := range getMasterArguments() {
		if value, ok := job.Annotations[item]; ok {
			command += fmt.Sprintf(" --%s %s", item, value)
		}
	}

	container := corev1.Container{
		Name:            "main",
		Image:           masterImage,
		ImagePullPolicy: defaultImagePullPolicy,
		Command:         []string{"/bin/bash", "-c", command},
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
	return podTemplate
}

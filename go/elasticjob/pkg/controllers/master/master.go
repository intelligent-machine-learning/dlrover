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
	"context"
	"fmt"
	"strconv"

	elasticv1alpha1 "github.com/intelligent-machine-learning/dlrover/go/elasticjob/api/v1alpha1"
	common "github.com/intelligent-machine-learning/dlrover/go/elasticjob/pkg/common"
	commonv1 "github.com/intelligent-machine-learning/dlrover/go/elasticjob/pkg/common/api/v1"
	logger "github.com/sirupsen/logrus"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	runtime_client "sigs.k8s.io/controller-runtime/pkg/client"
)

const (
	initMasterContainerCPU     = "1"
	initMasterContainerMemory  = "2Gi"
	initMasterContainerStorage = "2Gi"
	masterCommand              = "python -m dlrover.python.master.main"
	masterServicePort          = 50001
	initMasterIndex            = 0
	defaultImagePullPolicy     = "Always"
	envMasterAddrKey           = "DLROVER_MASTER_ADDR"
	envBrainServiceAddrKey     = "DLROVER_BRAIN_SERVICE_ADDR"
	defaultBrainServiceAddr    = "dlrover-brain.dlrover.svc.cluster.local:50001"
	envPodIP                   = "POD_IP"

	// ReplicaTypeJobMaster is the type for DLRover ElasticJob Master replica.
	ReplicaTypeJobMaster commonv1.ReplicaType = "dlrover-master"

	// supported arguments(should be supported in 'dlrover.python.master.args')
	pendingTimeoutArg      = "pending_timeout"
	pendingFailStrategyArg = "pending_fail_strategy"
	serviceType            = "service_type"
	preCheckOperatorsArg   = "pre_check_ops"
	preCheckBypassArg      = "pre_check_bypass"
)

// Manager generates a master pod object.
type Manager struct{}

func init() {
	common.ReplicaManagers[ReplicaTypeJobMaster] = &Manager{}
}

func (m *Manager) newJobMaster(
	job *elasticv1alpha1.ElasticJob, replicaIndex int,
) *corev1.Pod {
	masterName := newJobMasterName(job.Name)
	pod := common.NewPod(
		job, &job.Spec.ReplicaSpecs[ReplicaTypeJobMaster].Template, masterName,
	)
	pod.Labels[common.LabelReplicaTypeKey] = string(ReplicaTypeJobMaster)
	pod.Labels[common.LabelReplicaIndexKey] = fmt.Sprintf("%d", replicaIndex)
	if job.Spec.OptimizeMode == "cluster" {
		brainServiceAddr := defaultBrainServiceAddr
		if job.Spec.BrainService != "" {
			brainServiceAddr = job.Spec.BrainService
		}
		setBrainServiceIntoContainer(&pod.Spec.Containers[0], brainServiceAddr)
	}
	return pod
}

// ReconcilePods reconciles Pods of a job on a K8s cluster
func (m *Manager) ReconcilePods(
	client runtime_client.Client,
	job *elasticv1alpha1.ElasticJob,
	scalePlan *elasticv1alpha1.ScalePlan,
) error {
	master, _ := m.getMasterPod(client, job)
	if master != nil {
		logger.Warnf("Master exists")
		return nil
	}
	masterPod := m.newJobMaster(job, initMasterIndex)
	err := client.Create(context.Background(), masterPod)
	if err != nil {
		return err
	}
	service := m.newJobMasterService(job)
	err = client.Create(context.Background(), service)
	if err != nil {
		return err
	}
	return nil
}

// SyncJobState synchronize the job status by replicas
func (m *Manager) SyncJobState(client runtime_client.Client, job *elasticv1alpha1.ElasticJob) error {
	master, err := m.getMasterPod(client, job)
	if master == nil {
		logger.Warnf("Failed to get master, error : %v", err)
		return nil
	}
	masterIndex, _ := strconv.Atoi(master.Labels[common.LabelReplicaIndexKey])
	job.Status.ReplicaStatuses[ReplicaTypeJobMaster] = common.GetReplicaStatus([]corev1.Pod{*master})
	if master.Status.Phase == corev1.PodSucceeded {
		msg := fmt.Sprintf("job(%s/%s) successfully completed", job.Namespace, job.Name)
		if job.Status.CompletionTime == nil {
			now := metav1.Now()
			job.Status.CompletionTime = &now
		}
		common.UpdateStatus(&job.Status, commonv1.JobSucceeded, common.JobCreatedReason, msg)
	} else if master.Status.Phase == corev1.PodFailed {
		msg := fmt.Sprintf("job(%s/%s) has failed", job.Namespace, job.Name)
		reason := master.Status.Reason
		if reason == "" {
			reason = common.JobFailedReason
		}
		if job.Status.CompletionTime == nil {
			now := metav1.Now()
			job.Status.CompletionTime = &now
		}
		common.UpdateStatus(&job.Status, commonv1.JobFailed, reason, msg)
	} else if master.Status.Phase == corev1.PodPending && masterIndex == 0 {
		msg := fmt.Sprintf("job(%s/%s) is pending.", job.Namespace, job.Name)
		common.UpdateStatus(&job.Status, commonv1.JobPending, common.JobPendingReason, msg)
	} else if master.Status.Phase == corev1.PodRunning {
		if job.Status.Phase != commonv1.JobRunning {
			msg := fmt.Sprintf("job(%s/%s) is running.", job.Namespace, job.Name)
			common.UpdateStatus(&job.Status, commonv1.JobRunning, common.JobRunningReason, msg)
		}
	}
	return nil
}

// getMasterPod gets the master pod of a job from a cluster.
func (m *Manager) getMasterPod(client runtime_client.Client, job *elasticv1alpha1.ElasticJob) (*corev1.Pod, error) {
	pods, err := common.GetReplicaTypePods(client, job, ReplicaTypeJobMaster)
	if errors.IsNotFound(err) {
		return nil, err
	}
	if len(pods) == 0 {
		logger.Warnf("No master of job %s Found", job.Name)
		return nil, nil
	}

	return &pods[len(pods)-1], nil
}

func (m *Manager) newJobMasterService(job *elasticv1alpha1.ElasticJob) *corev1.Service {
	name := NewEasydlMasterServiceName(job.Name)
	selector := make(map[string]string)
	selector[common.LabelReplicaTypeKey] = string(ReplicaTypeJobMaster)
	service := common.NewService(job, name, masterServicePort, selector)
	return service
}

// NewEasydlMasterServiceName create a service name for Job master
func NewEasydlMasterServiceName(jobName string) string {
	return fmt.Sprintf("elasticjob-%s-%s", jobName, string(ReplicaTypeJobMaster))
}

// newJobMasterName create a name for Job master
func newJobMasterName(jobName string) string {
	return fmt.Sprintf("elasticjob-%s-%s", jobName, string(ReplicaTypeJobMaster))
}

// HandleFaultPods relaunches a new Pod if a pod is deleted or ignores
// the fault Pod if it fails with uncoverable errors.
func (m *Manager) HandleFaultPods(client runtime_client.Client, job *elasticv1alpha1.ElasticJob) error {
	curMaster, _ := m.getMasterPod(client, job)
	if curMaster == nil {
		newMasterPod := m.newJobMaster(job, initMasterIndex)
		logger.Infof("Master %s is deleted and relaunch a new one", newMasterPod.Name)
		err := client.Create(context.Background(), newMasterPod)
		if err != nil {
			return err
		}
	}
	if curMaster.DeletionTimestamp != nil {
		curIndex, err := strconv.Atoi(curMaster.Labels[common.LabelReplicaIndexKey])
		if err != nil {
			return err
		}
		newIndex := curIndex + 1
		newMasterPod := m.newJobMaster(job, newIndex)
		logger.Infof("Master %s is deleted and relaunch a new one %s", curMaster.Name, newMasterPod.Name)
		err = client.Create(context.Background(), newMasterPod)
		if err != nil {
			return err
		}
	}
	return nil
}

// SetMasterAddrIntoContainer sets the master service address into pod envs
func SetMasterAddrIntoContainer(container *corev1.Container, jobName string) {
	masterAddrEnv := newMasterAddrEnvVar(jobName)
	container.Env = append(container.Env, masterAddrEnv)
}

func newMasterAddrEnvVar(jobName string) corev1.EnvVar {
	masterServiceAddr := NewEasydlMasterServiceName(jobName)
	return corev1.EnvVar{
		Name:  envMasterAddrKey,
		Value: fmt.Sprintf("%s:%d", masterServiceAddr, masterServicePort),
	}
}

func setBrainServiceIntoContainer(container *corev1.Container, serviceAddr string) {
	envVar := corev1.EnvVar{
		Name:  envBrainServiceAddrKey,
		Value: serviceAddr,
	}
	container.Env = append(container.Env, envVar)
}

// StopRunningPods stops all running master Pods
func (m *Manager) StopRunningPods(
	client runtime_client.Client,
	job *elasticv1alpha1.ElasticJob,
) error {
	pod, err := m.getMasterPod(client, job)
	if pod == nil {
		logger.Warnf("Job %s: Failed to get master, error : %v", job.Name, err)
		return nil
	}
	if pod.Status.Phase == corev1.PodRunning || pod.Status.Phase == corev1.PodPending {
		common.DeletePod(client, pod)
	}
	return nil
}

func getMasterArguments() []string {
	return []string{pendingTimeoutArg, pendingFailStrategyArg, serviceType,
		preCheckOperatorsArg, preCheckBypassArg}
}

// NewMasterTemplateToJob sets configurations to the master template of a job.
func NewMasterTemplateToJob(job *elasticv1alpha1.ElasticJob, masterImage string) {
	var podTemplate *corev1.PodTemplateSpec
	if _, ok := job.Spec.ReplicaSpecs[ReplicaTypeJobMaster]; ok {
		podTemplate = &job.Spec.ReplicaSpecs[ReplicaTypeJobMaster].Template
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
	job.Spec.ReplicaSpecs[ReplicaTypeJobMaster] = &elasticv1alpha1.ReplicaSpec{
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

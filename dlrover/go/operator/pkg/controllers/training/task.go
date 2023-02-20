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

package training

import (
	"context"
	"fmt"
	elasticv1alpha1 "github.com/intelligent-machine-learning/easydl/dlrover/go/operator/api/v1alpha1"
	common "github.com/intelligent-machine-learning/easydl/dlrover/go/operator/pkg/common"
	commonv1 "github.com/intelligent-machine-learning/easydl/dlrover/go/operator/pkg/common/api/v1"
	master "github.com/intelligent-machine-learning/easydl/dlrover/go/operator/pkg/controllers/master"
	logger "github.com/sirupsen/logrus"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	runtime_client "sigs.k8s.io/controller-runtime/pkg/client"
	"sort"
	"strconv"
	"strings"
)

const (
	// ReplicaTypeWorker is the type for training worker replica.
	ReplicaTypeWorker commonv1.ReplicaType = "worker"

	// ReplicaTypePS is the type for training parameter server replica
	ReplicaTypePS commonv1.ReplicaType = "ps"

	// ReplicaTypeChief is the type for training chief replica of TensorFlow PS.
	ReplicaTypeChief commonv1.ReplicaType = "chief"

	// ReplicaTypeEvaluator is the type for elaluator replica
	ReplicaTypeEvaluator commonv1.ReplicaType = "evaluator"

	// LabelRestartCount is the count to relaunch failed nodes
	LabelRestartCount = "restart-count"

	// EnvTfConfigName is the environment variable name of TensorFlow cluster spec.
	EnvTfConfigName = "TF_CONFIG"

	// ServicePort is the port of service
	ServicePort int = 2222
)

// TaskManager generates Pods for task in a distributed PS job.
type TaskManager struct {
	taskType commonv1.ReplicaType
}

func init() {
	logger.Infof("init training task manager")
	common.ReplicaManagers[ReplicaTypeChief] = &TaskManager{
		taskType: ReplicaTypeChief,
	}
	common.ReplicaManagers[ReplicaTypeWorker] = &TaskManager{
		taskType: ReplicaTypeWorker,
	}
	common.ReplicaManagers[ReplicaTypeEvaluator] = &TaskManager{
		taskType: ReplicaTypeEvaluator,
	}
	common.ReplicaManagers[ReplicaTypePS] = &TaskManager{
		taskType: ReplicaTypePS,
	}
}

// SyncJobState synchronize the job status by replicas
func (m *TaskManager) SyncJobState(
	client runtime_client.Client,
	job *elasticv1alpha1.ElasticJob,
) error {
	taskPods, err := common.GetReplicaTypePods(client, job, m.taskType)
	if errors.IsNotFound(err) {
		logger.Warningf("No any Task %s found: %v", m.taskType, err)
		return nil
	}
	taskStatus := common.GetReplicaStatus(taskPods)
	if _, ok := job.Status.ReplicaStatuses[m.taskType]; !ok {
		job.Status.ReplicaStatuses[m.taskType] = taskStatus
	} else {
		job.Status.ReplicaStatuses[m.taskType].Pending = taskStatus.Pending
		job.Status.ReplicaStatuses[m.taskType].Active = taskStatus.Active
		job.Status.ReplicaStatuses[m.taskType].Failed = taskStatus.Failed
		job.Status.ReplicaStatuses[m.taskType].Succeeded = taskStatus.Succeeded
	}
	return nil
}

// ReconcilePods creates a Pod on a K8s cluster
func (m *TaskManager) ReconcilePods(
	client runtime_client.Client,
	job *elasticv1alpha1.ElasticJob,
	scalePlan *elasticv1alpha1.ScalePlan,
) error {
	status := m.getTaskStatus(job)
	aliveNum := int(status.Active + status.Pending + status.Succeeded)
	if resourceSpec, ok := scalePlan.Spec.ReplicaResourceSpecs[m.taskType]; ok {
		diffNum := resourceSpec.Replicas - aliveNum
		if diffNum > 0 {
			m.scaleUpReplicas(
				client, job, scalePlan, &resourceSpec.Resource, diffNum,
			)
		} else if diffNum < 0 {
			diffNum = -1 * diffNum
			m.scaleDownReplicas(client, job, diffNum)
		}
	}
	for _, podMeta := range scalePlan.Spec.CreatePods {
		if podMeta.Type == m.taskType {
			err := m.createPod(client, job, scalePlan, &podMeta)
			if err != nil {
				return err
			}
		}

	}
	for _, podMeta := range scalePlan.Spec.RemovePods {
		if podMeta.Type == m.taskType {
			err := m.removePod(client, &podMeta)
			if err != nil {
				return err
			}
		}
	}
	return nil
}

func (m *TaskManager) scaleUpReplicas(
	client runtime_client.Client,
	job *elasticv1alpha1.ElasticJob,
	scalePlan *elasticv1alpha1.ScalePlan,
	resource *corev1.ResourceList,
	upNum int,
) error {
	status := m.getTaskStatus(job)
	taskNum := m.getTotalTaskCount(status)
	for i := taskNum; i < taskNum+upNum; i++ {
		podService := m.newTaskServiceAddr(job.Name, i, ServicePort)
		podMeta := &elasticv1alpha1.PodMeta{
			Name:      m.newTaskName(job.Name, i),
			ID:        i,
			RankIndex: i,
			Type:      m.taskType,
			Service:   podService,
			Resource:  *resource,
		}
		err := m.createPod(client, job, scalePlan, podMeta)
		if err != nil {
			return err
		}
	}
	return nil
}

func (m *TaskManager) scaleDownReplicas(
	client runtime_client.Client,
	job *elasticv1alpha1.ElasticJob,
	downNum int,
) error {
	pods, err := common.GetReplicaTypePods(client, job, m.taskType)
	if errors.IsNotFound(err) {
		logger.Warningf("No any worker found: %v", err)
		return nil
	}
	alivePods := make(map[int]*corev1.Pod)
	PodIDs := []int{}
	for _, pod := range pods {
		if pod.Status.Phase == corev1.PodRunning || pod.Status.Phase == corev1.PodPending {
			PodID, _ := strconv.Atoi(pod.Labels[common.LabelReplicaIndexKey])
			PodIDs = append(PodIDs, PodID)
			alivePods[PodID] = &pod
		}
	}
	sort.Sort(sort.Reverse(sort.IntSlice(PodIDs)))
	for i := 0; i < downNum; i++ {
		common.DeletePod(client, alivePods[i])
	}

	return nil
}

func (m *TaskManager) newTask(
	job *elasticv1alpha1.ElasticJob,
	podMeta *elasticv1alpha1.PodMeta,
) *corev1.Pod {
	spec := job.Spec.ReplicaSpecs[m.taskType]
	if spec == nil && m.taskType == ReplicaTypeChief {
		spec = job.Spec.ReplicaSpecs[ReplicaTypeWorker]
		if spec != nil {
			logger.Infof("Use worker spec to create the chief")
		}
	}
	if spec == nil {
		logger.Errorf("No replica specification for %s", m.taskType)
		return nil
	}
	podTemplate := spec.Template.DeepCopy()
	podTemplate.Spec.Containers[0].Resources.Requests = podMeta.Resource
	podTemplate.Spec.Containers[0].Resources.Limits = podMeta.Resource
	pod := common.NewPod(job, podTemplate, podMeta.Name)
	master.SetMasterAddrIntoContainer(&pod.Spec.Containers[0], job.Name)
	pod.Labels[LabelRestartCount] = fmt.Sprintf("%d", spec.RestartCount)
	pod.Labels[common.LabelReplicaTypeKey] = string(m.taskType)
	pod.Labels[common.LabelReplicaIndexKey] = fmt.Sprintf("%d", podMeta.ID)
	pod.Labels[common.LabelRankIndexKey] = fmt.Sprintf("%d", podMeta.RankIndex)
	return pod
}

func (m *TaskManager) newTaskName(jobName string, taskIndex int) string {
	return fmt.Sprintf("%s-edljob-%s-%d", jobName, string(m.taskType), taskIndex)
}

func (m *TaskManager) newTaskServiceAddr(
	jobName string,
	taskIndex int,
	port int,
) string {
	return fmt.Sprintf("%s-edljob-%s-%d:%d", jobName, string(m.taskType), taskIndex, port)
}

func (m *TaskManager) getTaskStatus(
	job *elasticv1alpha1.ElasticJob,
) *commonv1.ReplicaStatus {
	replicaStatus, ok := job.Status.ReplicaStatuses[m.taskType]
	if !ok {
		return &commonv1.ReplicaStatus{}
	}
	return replicaStatus
}

func (m *TaskManager) getTotalTaskCount(taskStatus *commonv1.ReplicaStatus) int {
	if taskStatus == nil {
		return 0
	}
	return int(taskStatus.Active + taskStatus.Pending + taskStatus.Succeeded + taskStatus.Failed)
}

func (m *TaskManager) newServiceForPod(
	job *elasticv1alpha1.ElasticJob,
	podMeta *elasticv1alpha1.PodMeta,
) *corev1.Service {
	host := strings.Split(podMeta.Service, ":")[0]
	name := strings.Split(host, ".")[0]
	port, _ := strconv.Atoi(strings.Split(podMeta.Service, ":")[1])
	selector := make(map[string]string)
	selector[common.LabelReplicaTypeKey] = string(m.taskType)
	selector[common.LabelReplicaIndexKey] = fmt.Sprintf("%d", podMeta.ID)
	service := common.NewService(job, name, port, selector)
	return service
}

func (m *TaskManager) getAllTaskHosts(
	jobName string,
	totalTaskCount int,
	port int,
) []string {
	hosts := []string{}
	for i := 0; i < totalTaskCount; i++ {
		serviceName := m.newTaskServiceAddr(jobName, i, port)
		hosts = append(hosts, serviceName)
	}
	return hosts
}

// getPSClusterForPod gets a cluster definition of a PS training cluster
func (m *TaskManager) getPSClusterForPod(
	client runtime_client.Client,
	job *elasticv1alpha1.ElasticJob,
	scalePlan *elasticv1alpha1.ScalePlan,
	podMeta *elasticv1alpha1.PodMeta,
) SparseClusterSpec {
	cluster := SparseClusterSpec{}
	cluster.PS = scalePlan.Spec.PsHosts
	if status, ok := job.Status.ReplicaStatuses[ReplicaTypeChief]; ok {
		cluster.Chief = make(map[int]string)
		chiefManager := common.ReplicaManagers[ReplicaTypeChief].(*TaskManager)
		taskCount := chiefManager.getTotalTaskCount(status)
		if taskCount == 0 {
			taskCount = int(status.Initial) + scalePlan.Spec.ReplicaResourceSpecs[ReplicaTypeChief].Replicas
		}
		chiefHosts := chiefManager.getAllTaskHosts(job.Name, taskCount, ServicePort)
		if len(chiefHosts) == 1 {
			cluster.Chief[0] = chiefHosts[0]
		} else {
			logger.Errorf("The number of chief is not 1")
		}
	}
	if m.taskType == ReplicaTypePS {
		cluster.Worker = make(map[int]string)
		taskCounts := m.getTaskCounts(job, scalePlan)
		workerNum := taskCounts[ReplicaTypeWorker]
		for i := 0; i < int(workerNum); i++ {
			cluster.Worker[i] = fmt.Sprintf("%s-%s-%d:%d", job.Name, ReplicaTypeWorker, i, ServicePort)
		}
	}
	if m.taskType == ReplicaTypeChief {
		if cluster.Chief == nil {
			cluster.Chief = make(map[int]string)
		}
		cluster.Chief[podMeta.RankIndex] = podMeta.Service
	}
	if m.taskType == ReplicaTypeWorker {
		if cluster.Worker == nil {
			cluster.Worker = make(map[int]string)
		}
		cluster.Worker[podMeta.RankIndex] = podMeta.Service
	}
	if m.taskType == ReplicaTypeEvaluator {
		if cluster.Evaluator == nil {
			cluster.Evaluator = make(map[int]string)
		}
		cluster.Evaluator[podMeta.RankIndex] = podMeta.Service
	}
	return cluster
}

func (m *TaskManager) getTaskCounts(
	job *elasticv1alpha1.ElasticJob,
	scalePlan *elasticv1alpha1.ScalePlan,
) map[commonv1.ReplicaType]int32 {
	taskCounts := make(map[commonv1.ReplicaType]int32)
	for replicaType, replicaStatus := range job.Status.ReplicaStatuses {
		taskCounts[replicaType] = replicaStatus.Active + replicaStatus.Pending + replicaStatus.Succeeded
	}
	for replicaType, resourceSpec := range scalePlan.Spec.ReplicaResourceSpecs {
		taskCounts[replicaType] = int32(resourceSpec.Replicas)
	}
	return taskCounts
}

// HandleFaultPods processes fault Pods
func (m *TaskManager) HandleFaultPods(
	client runtime_client.Client, job *elasticv1alpha1.ElasticJob,
) error {
	return nil
}

func (m *TaskManager) createPod(
	client runtime_client.Client,
	job *elasticv1alpha1.ElasticJob,
	scalePlan *elasticv1alpha1.ScalePlan,
	podMeta *elasticv1alpha1.PodMeta,
) error {

	pod := m.newTask(job, podMeta)
	if job.Spec.DistributionStrategy == "ParameterServerStrategy" {
		cluster := m.getPSClusterForPod(client, job, scalePlan, podMeta)
		InsertTfConfigToEnv(
			&pod.Spec.Containers[0], cluster, podMeta.Type, podMeta.RankIndex,
		)
	}
	err := client.Create(context.Background(), pod)
	if err != nil {
		return err
	}
	service := m.newServiceForPod(job, podMeta)
	err = client.Create(context.Background(), service)
	if err != nil {
		return err
	}
	return nil
}

func (m *TaskManager) removePod(
	client runtime_client.Client,
	podMeta *elasticv1alpha1.PodMeta,
) error {
	pod := &corev1.Pod{}
	pod.Name = podMeta.Name
	return common.DeletePod(client, pod)
}

// StopRunningPods stops all running Pods
func (m *TaskManager) StopRunningPods(
	client runtime_client.Client,
	job *elasticv1alpha1.ElasticJob,
) error {
	taskPods, err := common.GetReplicaTypePods(client, job, m.taskType)
	if errors.IsNotFound(err) {
		logger.Warningf("No any Task %s found: %v", m.taskType, err)
		return nil
	}
	for _, pod := range taskPods {
		if pod.Status.Phase == corev1.PodRunning || pod.Status.Phase == corev1.PodPending {
			common.DeletePod(client, &pod)
		}
	}
	return nil
}

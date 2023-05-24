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
	"time"
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

	// PSServicePort is the port of service
	PSServicePort int = 2222

	// WorkerServicePort is the port of service
	WorkerServicePort int = 3333

	workerTypeEnvName   = "WORKER_TYPE"
	workerIDEnvName     = "WORKER_ID"
	workerRankEnvName   = "WORKER_RANK"
	workerNumEnvName    = "WORKER_NUM"
	rdzvEndpointEnvName = "RDZV_ENDPOINT"
	grpcEnableFork      = "GRPC_ENABLE_FORK_SUPPORT"
	podName             = "POD_NAME"
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

func newTaskName(jobName string, taskType string, taskIndex int) string {
	return fmt.Sprintf("%s-edljob-%s-%d", jobName, taskType, taskIndex)
}

func newTaskServiceAddr(
	jobName string,
	taskType string,
	taskIndex int,
	port int,
) string {
	taskName := newTaskName(jobName, taskType, taskIndex)
	return fmt.Sprintf("%s:%d", taskName, port)
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
	aliveNum := m.getAlivePodNum(client, job)
	if resourceSpec, ok := scalePlan.Spec.ReplicaResourceSpecs[m.taskType]; ok {
		diffNum := resourceSpec.Replicas - aliveNum
		logger.Infof("Job %s: Scale %s Pods with the number %d", job.Name, m.taskType, diffNum)
		if diffNum > 0 {
			err := m.scaleUpReplicas(
				client, job, scalePlan, &resourceSpec.Resource, diffNum,
			)
			if err != nil {
				return err
			}
		} else if diffNum < 0 {
			diffNum = -1 * diffNum
			err := m.scaleDownReplicas(client, job, diffNum)
			if err != nil {
				return err
			}
		}
	}
	for _, podMeta := range scalePlan.Spec.CreatePods {
		if podMeta.Type == m.taskType {
			logger.Infof("Create %s Pod with metas %v", m.taskType, podMeta)
			err := m.createPod(client, job, scalePlan, &podMeta)
			if err != nil {
				return err
			}
		}

	}
	for _, podMeta := range scalePlan.Spec.RemovePods {
		if podMeta.Type == m.taskType {
			err := m.removePod(client, &podMeta, job.Namespace)
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
	startTaskID := m.getMaxReplicaID(client, job) + 1
	for i := startTaskID; i < startTaskID+upNum; i++ {
		logger.Infof("Job %s: Create %d %s", job.Name, i, m.taskType)
		port := WorkerServicePort
		if m.taskType == ReplicaTypePS {
			port = PSServicePort
		}
		podService := newTaskServiceAddr(
			job.Name, string(m.taskType), i, port,
		)
		podMeta := &elasticv1alpha1.PodMeta{
			Name:      newTaskName(job.Name, string(m.taskType), i),
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

func (m *TaskManager) getAlivePodNum(
	client runtime_client.Client,
	job *elasticv1alpha1.ElasticJob,
) int {
	aliveCount := 0
	for i := 0; i < 3; i++ {
		pods, err := common.GetReplicaTypePods(client, job, m.taskType)
		if err == nil {
			for _, pod := range pods {
				phase := pod.Status.Phase
				if phase == corev1.PodRunning || phase == corev1.PodPending || phase == corev1.PodSucceeded {
					aliveCount = aliveCount + 1
				}
			}
			break
		} else {
			time.Sleep(200 * time.Millisecond)
		}
	}
	return aliveCount
}

func (m *TaskManager) getMaxReplicaID(
	client runtime_client.Client,
	job *elasticv1alpha1.ElasticJob,
) int {
	pods, err := common.GetReplicaTypePods(client, job, m.taskType)
	if err != nil {
		return -1
	}
	maxID := -1
	for _, pod := range pods {
		PodID, _ := strconv.Atoi(pod.Labels[common.LabelReplicaIndexKey])
		if PodID > maxID {
			maxID = PodID
		}
	}
	return maxID
}

func (m *TaskManager) scaleDownReplicas(
	client runtime_client.Client,
	job *elasticv1alpha1.ElasticJob,
	downNum int,
) error {
	logger.Infof("Job %s: Scale down %d %s Pods", job.Name, downNum, m.taskType)
	pods, err := common.GetReplicaTypePods(client, job, m.taskType)
	if errors.IsNotFound(err) {
		logger.Warningf("No any worker found: %v", err)
		return nil
	}
	alivePods := make(map[int]corev1.Pod)
	PodIDs := []int{}
	for _, pod := range pods {
		if pod.Status.Phase == corev1.PodRunning || pod.Status.Phase == corev1.PodPending {
			PodID, _ := strconv.Atoi(pod.Labels[common.LabelReplicaIndexKey])
			PodIDs = append(PodIDs, PodID)
			alivePods[PodID] = pod
		}
	}
	sort.Sort(sort.Reverse(sort.IntSlice(PodIDs)))
	for i := 0; i < downNum; i++ {
		pod := alivePods[PodIDs[i]]
		logger.Infof("Delete Pod %s", pod.Name)
		err := common.DeletePod(client, &pod)
		if err != nil {
			return err
		}
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

	container := &pod.Spec.Containers[0]
	workerTypeEnv := corev1.EnvVar{
		Name:  workerTypeEnvName,
		Value: string(m.taskType),
	}
	container.Env = append(container.Env, workerTypeEnv)
	workerIDEnv := corev1.EnvVar{
		Name:  workerIDEnvName,
		Value: fmt.Sprintf("%d", podMeta.ID),
	}
	container.Env = append(container.Env, workerIDEnv)
	rankIDEnv := corev1.EnvVar{
		Name:  workerRankEnvName,
		Value: fmt.Sprintf("%d", podMeta.RankIndex),
	}
	container.Env = append(container.Env, rankIDEnv)
	grpcEnableForkEnv := corev1.EnvVar{
		Name:  grpcEnableFork,
		Value: "False",
	}
	container.Env = append(container.Env, grpcEnableForkEnv)
	podNameEnv := corev1.EnvVar{Name: podName}
	podNameEnv.ValueFrom = &corev1.EnvVarSource{
		FieldRef: &corev1.ObjectFieldSelector{
			APIVersion: "v1",
			FieldPath:  "metadata.name",
		},
	}
	container.Env = append(container.Env, podNameEnv)
	if m.taskType == ReplicaTypeWorker {
		workerNumEnv := corev1.EnvVar{
			Name:  workerNumEnvName,
			Value: fmt.Sprintf("%d", spec.Replicas),
		}
		container.Env = append(container.Env, workerNumEnv)
	}
	return pod
}

func (m *TaskManager) setAllreduceEnv(
	client runtime_client.Client,
	job *elasticv1alpha1.ElasticJob,
	container *corev1.Container,
) error {
	worker0Name := newTaskName(job.Name, string(m.taskType), 0)
	worker0, err := common.GetPod(client, job.Namespace, worker0Name)
	rdzvEndpointEnv := corev1.EnvVar{Name: rdzvEndpointEnvName}
	if err != nil {
		rdzvEndpointEnv.ValueFrom = &corev1.EnvVarSource{
			FieldRef: &corev1.ObjectFieldSelector{
				APIVersion: "v1",
				FieldPath:  "status.podIP",
			},
		}
	} else {
		rdzvEndpointEnv.Value = worker0.Status.PodIP
	}
	container.Env = append(container.Env, rdzvEndpointEnv)
	return nil
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
	selector[common.LabelRankIndexKey] = fmt.Sprintf("%d", podMeta.RankIndex)
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
		serviceName := newTaskServiceAddr(jobName, string(m.taskType), i, port)
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
	chiefManager := common.ReplicaManagers[ReplicaTypeChief].(*TaskManager)
	chiefHosts := chiefManager.getAllTaskHosts(job.Name, 1, WorkerServicePort)
	cluster.Chief = make(map[int]string)
	if len(chiefHosts) == 1 {
		cluster.Chief[0] = chiefHosts[0]
	} else {
		logger.Errorf("Job %s: The number of chief is not 1", job.Name)
	}
	if m.taskType == ReplicaTypePS {
		cluster.Worker = make(map[int]string)
		taskCounts := m.getTaskCounts(job, scalePlan)
		workerNum := taskCounts[ReplicaTypeWorker]
		for i := 0; i < int(workerNum); i++ {
			cluster.Worker[i] = newTaskServiceAddr(
				job.Name, string(ReplicaTypeWorker), i, WorkerServicePort,
			)
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
		for i := 0; i <= podMeta.RankIndex; i++ {
			cluster.Worker[i] = newTaskServiceAddr(
				job.Name, string(ReplicaTypeWorker), i, WorkerServicePort,
			)
		}
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
	} else if job.Spec.DistributionStrategy == "AllreduceStrategy" {
		m.setAllreduceEnv(client, job, &pod.Spec.Containers[0])
	}
	err := client.Create(context.Background(), pod)
	if errors.IsAlreadyExists(err) {
		logger.Infof("Pod %s is already exists.", pod.Name)
		return nil
	} else if err != nil {
		logger.Infof("Job %s: Fail to create Pod %s, %v", job.Name, pod.Name, err)
		return err
	}
	service := m.newServiceForPod(job, podMeta)
	err = client.Create(context.Background(), service)
	if errors.IsAlreadyExists(err) {
		err = client.Update(context.Background(), service)
		if err != nil {
			logger.Infof("Job %s: Fail to update service %s, %v", job.Name, service.Name, err)
			return err
		}
	} else if err != nil {
		logger.Infof("Job %s: Fail to create service %s, %v", job.Name, service.Name, err)
		return err
	}
	return nil
}

func (m *TaskManager) removePod(
	client runtime_client.Client,
	podMeta *elasticv1alpha1.PodMeta,
	namespace string,
) error {
	pod := &corev1.Pod{}
	pod.Name = podMeta.Name
	pod.Namespace = namespace
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

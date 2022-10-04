package psstrategy

import (
	"encoding/json"
	"fmt"
	elasticv1alpha1 "github.com/intelligent-machine-learning/easydl/operator/api/v1alpha1"
	commonv1 "github.com/intelligent-machine-learning/easydl/operator/pkg/common/api/v1"
	controllers "github.com/intelligent-machine-learning/easydl/operator/pkg/controllers"
	logger "github.com/sirupsen/logrus"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	runtime_client "sigs.k8s.io/controller-runtime/pkg/client"
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

	maxWorkerCount = 100
)

// TaskSpec is the specification for a task (PS or worker) of the ElasticJob using
// ParameterServerStrategy.
type TaskSpec struct {
	Type  commonv1.ReplicaType `json:"type"`
	Index int                  `json:"index"`
}

// SparseClusterSpec enables a server to be configured without needing to know
// the identity of (for example) all other worker tasks.
// https://www.tensorflow.org/api_docs/python/tf/train/ClusterSpec
type SparseClusterSpec struct {
	Worker    map[int]string `json:"worker,omitempty"`
	PS        []string       `json:"ps"`
	Chief     map[int]string `json:"chief"`
	Evaluator map[int]string `json:"evaluator,omitempty"`
}

// SparseTFConfig is a struct representing the distributed TensorFlow config.
type SparseTFConfig struct {
	// Cluster represents a TensorFlow ClusterSpec.
	// See: https://www.tensorflow.org/api_docs/python/tf/train/ClusterSpec
	Cluster SparseClusterSpec `json:"cluster"`
	Task    TaskSpec          `json:"task"`
}

// PSTaskManager generates Pods for task in a distributed PS job.
type PSTaskManager struct {
	controllers.PodManager

	taskType commonv1.ReplicaType
}

// SyncJobState synchronize the job status by replicas
func (m *PSTaskManager) SyncJobState(
	r *controllers.ElasticJobReconciler,
	job *elasticv1alpha1.ElasticJob,
) error {
	taskPods, err := m.GetReplicaTypePods(r.Client, job, m.taskType)
	for _, pod := range taskPods {
		logger.Infof("Pod :%s", pod.Name)
	}
	if errors.IsNotFound(err) {
		logger.Warningf("No any Task %s found: %v", m.taskType, err)
		return nil
	}
	taskStatus := m.GetReplicaStatus(taskPods)
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

func (m *PSTaskManager) insertCommonLabels(
	labels map[string]string, taskIndex int,
) {
	labels[controllers.LabelReplicaTypeKey] = string(m.taskType)
	labels[controllers.LabelReplicaIndexKey] = fmt.Sprintf("%d", taskIndex)
}

func (m *PSTaskManager) newTask(
	job *elasticv1alpha1.ElasticJob,
	taskIndex int,
) *corev1.Pod {
	spec, ok := job.Spec.ReplicaSpecs[m.taskType]
	if !ok {
		logger.Infof("No ReplicaSpec for %s", m.taskType)
		return nil
	}
	name := m.newTaskName(job.Name, taskIndex)
	pod := m.NewPod(job, &spec.Template, name)
	pod.Labels[LabelRestartCount] = fmt.Sprintf("%d", spec.RestartCount)
	m.insertCommonLabels(pod.Labels, taskIndex)
	return pod
}

func (m *PSTaskManager) newTaskName(jobName string, taskIndex int) string {
	return fmt.Sprintf("%s-%s-%d", jobName, string(m.taskType), taskIndex)
}

func (m *PSTaskManager) newTaskServiceAddr(
	jobName string,
	taskIndex int,
	port int,
) string {
	return fmt.Sprintf("%s-%s-%d:%d", jobName, string(m.taskType), taskIndex, port)
}

func (m *PSTaskManager) getTaskStatus(
	job *elasticv1alpha1.ElasticJob,
) *commonv1.ReplicaStatus {
	replicaStatus, ok := job.Status.ReplicaStatuses[m.taskType]
	if !ok {
		return &commonv1.ReplicaStatus{}
	}
	return replicaStatus
}

func (m *PSTaskManager) getTotalTaskCount(taskStatus *commonv1.ReplicaStatus) int {
	return int(taskStatus.Active + taskStatus.Pending + taskStatus.Succeeded + taskStatus.Failed)
}

func (m *PSTaskManager) newTaskService(
	job *elasticv1alpha1.ElasticJob,
	taskIndex int,
	servicePort int,
) *corev1.Service {
	name := m.newTaskName(job.Name, taskIndex)
	selector := make(map[string]string)
	m.insertCommonLabels(selector, taskIndex)
	service := m.NewService(job, name, servicePort, selector)
	return service
}

func (m *PSTaskManager) getAllTaskHosts(
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

func (m *PSTaskManager) getPSCluster(
	client runtime_client.Client,
	job *elasticv1alpha1.ElasticJob,
) SparseClusterSpec {
	cluster := SparseClusterSpec{}
	psHosts := m.getPSHosts(job, client)
	if len(psHosts) > 0 {
		cluster.PS = psHosts
	}
	if status, ok := job.Status.ReplicaStatuses[ReplicaTypeChief]; ok {
		cluster.Chief = make(map[int]string)
		chiefManager := controllers.ReplicaManagers[ReplicaTypeChief].(*ChiefManager)
		taskCount := chiefManager.getTotalTaskCount(status)
		if taskCount == 0 {
			taskCount = int(status.Initial)
		}
		chiefHosts := chiefManager.getAllTaskHosts(job.Name, taskCount, chiefServicePort)
		if len(chiefHosts) > 0 {
			cluster.Chief[0] = chiefHosts[0]
		}
	}
	if m.taskType == ReplicaTypePS {
		cluster.Worker = make(map[int]string)
		for i := 0; i < maxWorkerCount; i++ {
			cluster.Worker[i] = fmt.Sprintf("%s-%s-%d:%d", job.Name, ReplicaTypeWorker, i, workerServicePort)
		}
	}
	return cluster
}

func (m *PSTaskManager) getPSHosts(
	job *elasticv1alpha1.ElasticJob,
	client runtime_client.Client,
) []string {
	status := job.Status.ReplicaStatuses[ReplicaTypePS]
	psManager := controllers.ReplicaManagers[ReplicaTypePS].(*PSManager)
	taskCount := psManager.getTotalTaskCount(status)
	var psHosts []string
	if taskCount == 0 {
		taskCount = int(status.Initial)
		psHosts = psManager.getAllTaskHosts(job.Name, taskCount, psServicePort)
	} else {
		psPods, _ := psManager.GetReplicaTypePods(client, job, psManager.taskType)
		psHosts = psManager.getAllPSHosts(psPods, job.Name)
	}
	return psHosts
}

func (m *PSTaskManager) genTFConfigEnv(cluster SparseClusterSpec, taskIndex int) (corev1.EnvVar, error) {
	tfConfig := SparseTFConfig{
		Cluster: cluster,
		Task: TaskSpec{
			Type:  m.taskType,
			Index: taskIndex,
		},
	}
	tfConfigJSONByteSlice, err := json.Marshal(tfConfig)
	tfConfigStr := string(tfConfigJSONByteSlice)
	return corev1.EnvVar{
		Name:  EnvTfConfigName,
		Value: tfConfigStr,
	}, err
}

func (m *PSTaskManager) insertTfConfigToEnv(container *corev1.Container, cluster SparseClusterSpec, taskIndex int) {
	tfConfigEnv, err := m.genTFConfigEnv(cluster, taskIndex)
	if err != nil {
		logger.Infof("Failed to get TFCONFIG %v", err)
		return
	}
	container.Env = append(container.Env, tfConfigEnv)
}

// HandleFaultPods relaunches a new Pod if a pod is deleted or ignores
// the fault Pod if it fails with uncoverable errors.
func (m *PSTaskManager) HandleFaultPods(
	r *controllers.ElasticJobReconciler, job *elasticv1alpha1.ElasticJob,
) error {
	replicaPods, err := m.GetReplicaTypePods(r.Client, job, m.taskType)
	if err != nil {
		return err
	}
	for _, pod := range(replicaPods){
		if pod.DeletionTimestamp != nil {
			logger.Infof("Pod %s is deleted and will be relaunched", pod.Name) 
			totalReplicaCount := m.getTotalTaskCount(job.Status.ReplicaStatuses[m.taskType])
			pod.Name = m.newTaskName(job.Name, totalReplicaCount)

		}else if pod.Status.Phase == corev1.PodFailed {
			if len(pod.Status.ContainerStatuses) > 0 && pod.Status.ContainerStatuses[0].State.Terminated != nil{
				terminated := pod.Status.ContainerStatuses[0].State.Terminated
				if terminated.Reason == commonv1.ReasonOOMKilled {
					if terminated.ExitCode == commonv1.FatalExitCode {
						logger.Infof("Pod %s fails", pod.Name)
					}
				}else {
					logger.Infof("Pod %s OOM", pod.Name)
				}
			}
		}
	}
	return nil
}


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
)

// TaskSpec is the specification for a task (PS or worker) of the ElasticJob using
// ParameterServerStrategy.
type TaskSpec struct {
	Type  commonv1.ReplicaType `json:"type"`
	Index int                  `json:"index"`
}

// ClusterSpec represents a cluster TensorFlow specification.
// https://www.tensorflow.org/deploy/distributed#create_a_tftrainclusterspec_to_describe_the_cluster
// It is a map from job names to network addresses.
type ClusterSpec map[commonv1.ReplicaType][]string

// TFConfig is a struct representing the distributed TensorFlow config.
// This struct is turned into an environment variable TF_CONFIG
// which is used by TensorFlow processes to configure themselves.
// https://www.tensorflow.org/api_docs/python/tf/estimator/RunConfig#methods
// https://cloud.google.com/ml-engine/docs/tensorflow/distributed-training-details
type TFConfig struct {
	// Cluster represents a TensorFlow ClusterSpec.
	// See: https://www.tensorflow.org/api_docs/python/tf/train/ClusterSpec
	Cluster ClusterSpec `json:"cluster"`
	Task    TaskSpec    `json:"task"`
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
	if errors.IsNotFound(err) {
		logger.Warningf("No any Task %s found: %v", m.taskType, err)
		return nil
	}
	taskStatus := m.GetReplicaStatus(taskPods)
	job.Status.ReplicaStatuses[m.taskType] = taskStatus
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
	taskStatus *commonv1.ReplicaStatus,
	port int,
) []string {
	totalTaskCount := m.getTotalTaskCount(taskStatus)
	hosts := []string{}
	for i := 0; i < totalTaskCount; i++ {
		chiefServiceName := m.newTaskServiceAddr(jobName, i, port)
		hosts = append(hosts, chiefServiceName)
	}
	return hosts
}

func (m *PSTaskManager) getPSCluster(
	client runtime_client.Client,
	job *elasticv1alpha1.ElasticJob,
) ClusterSpec {
	cluster := ClusterSpec{}
	if status, ok := job.Status.ReplicaStatuses[ReplicaTypeChief]; ok {
		chiefManager := controllers.ReplicaManagers[ReplicaTypeChief].(*ChiefManager)
		chiefHosts := chiefManager.getAllTaskHosts(job.Name, status, chiefServicePort)
		if len(chiefHosts) > 0 {
			cluster[ReplicaTypeChief] = chiefHosts
		}
	}

	if status, ok := job.Status.ReplicaStatuses[ReplicaTypeWorker]; ok {
		workerManager := controllers.ReplicaManagers[ReplicaTypeWorker].(*WorkerManager)
		workerHosts := workerManager.getAllTaskHosts(job.Name, status, workerServicePort)
		if len(workerHosts) > 0 {
			cluster[ReplicaTypeWorker] = workerHosts
		}
	}

	if status, ok := job.Status.ReplicaStatuses[ReplicaTypeEvaluator]; ok {
		evaluatorManager := controllers.ReplicaManagers[ReplicaTypeEvaluator].(*EvaluatorManager)
		evaluatorHosts := evaluatorManager.getAllTaskHosts(job.Name, status, evaluatorServicePort)
		if len(evaluatorHosts) > 0 {
			cluster[ReplicaTypeEvaluator] = evaluatorHosts
		}
	}

	if status, ok := job.Status.ReplicaStatuses[ReplicaTypePS]; ok && status.Active+status.Pending > 0 {
		psManager := controllers.ReplicaManagers[ReplicaTypePS].(*PSManager)
		psPods, _ := psManager.GetReplicaTypePods(client, job, psManager.taskType)
		psHosts := psManager.getAllPSHosts(psPods, job.Name)
		if len(psHosts) > 0 {
			cluster[ReplicaTypePS] = psHosts
		}
	}

	return cluster
}

func (m *PSTaskManager) genTFConfigEnv(cluster ClusterSpec, taskIndex int) (corev1.EnvVar, error) {
	tfConfig := TFConfig{
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

func (m *PSTaskManager) insertTfConfigToEnv(container *corev1.Container, cluster ClusterSpec, taskIndex int) {
	tfConfigEnv, err := m.genTFConfigEnv(cluster, taskIndex)
	if err != nil {
		logger.Infof("Failed to get TFCONFIG %v", err)
		return
	}
	container.Env = append(container.Env, tfConfigEnv)
}

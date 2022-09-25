package psstrategy

import (
	"fmt"
	elasticv1alpha1 "github.com/intelligent-machine-learning/easydl/operator/api/v1alpha1"
	commonv1 "github.com/intelligent-machine-learning/easydl/operator/pkg/common/api/v1"
	controllers "github.com/intelligent-machine-learning/easydl/operator/pkg/controllers"
	logger "github.com/sirupsen/logrus"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
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
)

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
	taskPods, err := m.GetReplicaTypePods(r, job, m.taskType)
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

func (m *PSTaskManager) newTask(job *elasticv1alpha1.ElasticJob, taskIndex int) *corev1.Pod {
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

func (m *PSTaskManager) newTaskServiceAddr(jobName string, taskIndex int, port int) string {
	return fmt.Sprintf("%s-%s-%d:%d", jobName, string(m.taskType), taskIndex, port)
}

func (m *PSTaskManager) getTaskStatus(job *elasticv1alpha1.ElasticJob) *commonv1.ReplicaStatus {
	replicaStatus, ok := job.Status.ReplicaStatuses[m.taskType]
	if !ok {
		return &commonv1.ReplicaStatus{}
	}
	return replicaStatus
}

func (m *PSTaskManager) getTotalTaskCount(taskStatus *commonv1.ReplicaStatus) int {
	return int(taskStatus.Active + taskStatus.Pending + taskStatus.Succeeded + taskStatus.Failed)
}

func (m *PSTaskManager) newTaskService(job *elasticv1alpha1.ElasticJob, taskIndex int, servicePort int) *corev1.Service {
	name := m.newTaskName(job.Name, taskIndex)
	selector := make(map[string]string)
	m.insertCommonLabels(selector, taskIndex)
	service := m.NewService(job, name, servicePort, selector)
	return service
}

func (m *PSTaskManager) getAllTaskHost(jobName string, taskStatus *commonv1.ReplicaStatus, port int) []string {
	totalTaskCount := m.getTotalTaskCount(taskStatus)
	hosts := []string{}
	for i := 0; i < totalTaskCount; i++ {
		chiefServiceName := m.newTaskServiceAddr(jobName, i, port)
		hosts = append(hosts, chiefServiceName)
	}
	return hosts
}

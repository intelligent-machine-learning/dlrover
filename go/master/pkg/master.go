package master

import (
	"time"

	elasticjob "github.com/intelligent-machine-learning/dlrover/go/elasticjob/api/v1alpha1"
	"github.com/intelligent-machine-learning/dlrover/go/master/pkg/kubernetes"
	logger "github.com/sirupsen/logrus"
)

type JobMaster struct {
	Namespace string
	JobName   string
	K8sClient *kubernetes.K8sClient
	Job       *elasticjob.ElasticJob
}

func NewJobMaster(namespace string, jobName string, k8sClient *kubernetes.K8sClient) *JobMaster {
	master := &JobMaster{
		Namespace: namespace,
		JobName:   jobName,
	}
	if k8sClient != nil {
		job := kubernetes.GetElasticJobInstance(k8sClient, namespace, jobName)
		master.K8sClient = k8sClient
		master.Job = job
	}
	logger.Infof("create a master of job %s.", jobName)
	return master
}

func (master *JobMaster) Run() {
	time.Sleep(10 * time.Hour)
}

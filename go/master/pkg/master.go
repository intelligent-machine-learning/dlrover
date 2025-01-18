// Copyright 2025 The DLRover Authors. All rights reserved.
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

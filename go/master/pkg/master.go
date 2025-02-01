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
	"context"
	"time"

	"github.com/intelligent-machine-learning/dlrover/go/master/pkg/common"
	"github.com/intelligent-machine-learning/dlrover/go/master/pkg/jobmanager"
	"github.com/intelligent-machine-learning/dlrover/go/master/pkg/kubeutils"
	logger "github.com/sirupsen/logrus"
)

// JobMaster is the master of an elasticjob.
type JobMaster struct {
	jobContext *common.JobContext
	jobManager jobmanager.JobManager
}

// NewJobMaster creates the master for an elasticjob.
func NewJobMaster(namespace string, jobName string) *JobMaster {
	master := &JobMaster{}
	if kubeutils.GlobalK8sClient != nil {
		elasticjob := kubeutils.GetElasticJobInstance(jobName)
		master.jobManager = jobmanager.NewJobManager(elasticjob)
	}
	master.jobContext = common.NewJobContext(namespace, jobName)
	logger.Infof("create a master of job %s.", jobName)
	return master
}

// Run starts the master instance.
func (master *JobMaster) Run() {
	ctx, cancel := context.WithCancel(context.Background())
	if master.jobManager != nil {
		master.jobManager.Start(ctx, master.jobContext)
	}
	defer cancel()
	time.Sleep(10 * time.Hour)
}

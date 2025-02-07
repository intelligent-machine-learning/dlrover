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

package jobmanager

import (
	"context"

	elasticjobv1 "github.com/intelligent-machine-learning/dlrover/go/elasticjob/api/v1alpha1"
	"github.com/intelligent-machine-learning/dlrover/go/master/pkg/batchscheduler"
	"github.com/intelligent-machine-learning/dlrover/go/master/pkg/common"
)

// PyTorchJobManager is the lifecycle manager of a PyTorch distributed training job.
type PyTorchJobManager struct {
	replicaSchedulers map[string]batchscheduler.BatchScheduler
}

// NewPyTorchJobManager creates PyTorch distributed training job manager.
func NewPyTorchJobManager(elasticJob *elasticjobv1.ElasticJob) *PyTorchJobManager {
	schedulers := make(map[string]batchscheduler.BatchScheduler)
	for replicaType, spec := range elasticJob.Spec.ReplicaSpecs {
		scheduler := batchscheduler.NewBatchScheduler(spec.BatchScheduler)
		schedulers[string(replicaType)] = scheduler
	}
	return &PyTorchJobManager{
		replicaSchedulers: schedulers,
	}
}

// Start starts the modules of the job manager.
func (jobManager *PyTorchJobManager) Start(ctx context.Context, jobContext *common.JobContext) {
	for _, scheduler := range jobManager.replicaSchedulers {
		scheduler.Start(ctx, jobContext)
	}
}

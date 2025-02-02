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
	"github.com/intelligent-machine-learning/dlrover/go/master/pkg/common"
)

// JobManager is the interface to manager job lifecycle.
type JobManager interface {
	Start(ctx context.Context, jobContext *common.JobContext)
}

// NewJobManager creates a job manager.
func NewJobManager(elasticJob *elasticjobv1.ElasticJob) JobManager {
	if elasticJob.Spec.DistributionStrategy == "pytorch" {
		return NewPyTorchJobManager(elasticJob)
	}
	return nil
}

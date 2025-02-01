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

package batchscheduler

import (
	elasticjob "github.com/intelligent-machine-learning/dlrover/go/elasticjob/api/v1alpha1"
	commonv1 "github.com/intelligent-machine-learning/dlrover/go/elasticjob/pkg/common/api/v1"
	"github.com/intelligent-machine-learning/dlrover/go/master/pkg/kubeutils"
)

// BatchScheduler creates/updates/deletes the batch pods of an elastic job.
type BatchScheduler interface {
	DoScheduling(*SchedulingPlan)
}

// SchedulingPlan is the scheduling plan to notify the scheduler CURD pods.
type SchedulingPlan struct {
	// ReplicaSpecs is a map which contains the replica specification to create Pods.
	ReplicaSpecs map[commonv1.ReplicaType]*commonv1.ReplicaSpec

	// CreatedPods are Pods to be created.
	CreatedPods []*kubeutils.PodConfig

	// RemovedPods are Pods to be removed
	RemovedPods []*kubeutils.PodConfig

	// OwnerJob specifies a job to scale.
	OwnerJob *elasticjob.ElasticJob
}

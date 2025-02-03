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
	"context"

	"github.com/intelligent-machine-learning/dlrover/go/master/pkg/common"
	"github.com/intelligent-machine-learning/dlrover/go/master/pkg/kubeutils"
	logger "github.com/sirupsen/logrus"
	"k8s.io/apimachinery/pkg/api/errors"
)

// ElasticScheduler launches pods without waiting for all resouces of pod are ready
type ElasticScheduler struct {
	KubeScheduler
	SchedulerName string
}

// NewElasticScheduler creates an elastic scheduler.
func NewElasticScheduler() *ElasticScheduler {
	return &ElasticScheduler{
		KubeScheduler: KubeScheduler{
			toCreatePods: common.NewQueue(),
		},
		SchedulerName: "elastic",
	}
}

// Start starts a routine to launch Pods.
func (scheduler *ElasticScheduler) Start(ctx context.Context, jobContext *common.JobContext) {
	go scheduler.LoopToLaunchPods(ctx)
}

// DoScheduling creates/updates/deletes pods
func (scheduler *ElasticScheduler) DoScheduling(jobContext *common.JobContext, plan *SchedulingPlan) {
	for replicaType, spec := range plan.ReplicaSpecs {
		for i := int32(0); i < spec.Replicas; i++ {
			replicaConfig := &kubeutils.ReplicaConfig{
				Type:   string(replicaType),
				ID:     i,
				Number: spec.Replicas,
				Rank:   i,
			}
			podConfig := &kubeutils.PodConfig{
				Replica:      replicaConfig,
				TemplateSpec: spec.Template.DeepCopy(),
			}
			pod := kubeutils.BuildPod(jobContext, podConfig, plan.OwnerJob)
			scheduler.toCreatePods.PushBack(pod)
		}
	}
	for _, podConfig := range plan.CreatedPods {
		pod := kubeutils.BuildPod(jobContext, podConfig, plan.OwnerJob)
		scheduler.toCreatePods.PushBack(pod)
	}
	for _, name := range plan.RemovedPods {
		err := kubeutils.GlobalK8sClient.RemovePod(name)
		if errors.IsNotFound(err) {
			logger.Infof("The Pod %s has been removed", name)
		} else {
			logger.Warnf("Fail to remove the pod %s, err = %v", name, err)
		}
	}
}

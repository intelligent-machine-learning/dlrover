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
	"time"

	elasticjobv1 "github.com/intelligent-machine-learning/dlrover/go/elasticjob/api/v1alpha1"
	commonv1 "github.com/intelligent-machine-learning/dlrover/go/elasticjob/pkg/common/api/v1"
	"github.com/intelligent-machine-learning/dlrover/go/master/pkg/common"
	"github.com/intelligent-machine-learning/dlrover/go/master/pkg/kubeutils"
	logger "github.com/sirupsen/logrus"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
)

// BatchScheduler creates/updates/deletes the batch pods of an elastic job.
type BatchScheduler interface {
	Start(ctx context.Context, jobContext *common.JobContext)
	DoScheduling(jobContext *common.JobContext, plan *SchedulingPlan)
}

// SchedulingPlan is the scheduling plan to notify the scheduler CURD pods.
type SchedulingPlan struct {
	// ReplicaSpecs is a map which contains the replica specification to create Pods.
	ReplicaSpecs map[commonv1.ReplicaType]*commonv1.ReplicaSpec

	// CreatedPods are Pods to be created.
	CreatedPods []*kubeutils.PodConfig

	// RemovedPods are Pods to be removed
	RemovedPods []string

	// OwnerJob specifies a job to scale.
	OwnerJob *elasticjobv1.ElasticJob
}

// KubeScheduler is the base scheduler to create/update/remove pods.
type KubeScheduler struct {
	toCreatePods *common.Queue
}

// NewBatchScheduler creates a batch scheduler according to the scheduler name.
func NewBatchScheduler(schedulerName string) BatchScheduler {
	if schedulerName == "elastic" || schedulerName == "" {
		scheduler := NewElasticScheduler()
		return scheduler
	}
	return nil
}

// LoopToLaunchPods launches pods from the pod queue.
func (scheduler *KubeScheduler) LoopToLaunchPods(ctx context.Context) {
	for {
		select {
		case <-ctx.Done():
			logger.Infof("The loop to launch Pod exists.")
		default:
			for scheduler.toCreatePods.Len() > 0 {
				pod := scheduler.toCreatePods.PopFront().(*corev1.Pod)
				err := kubeutils.GlobalK8sClient.CreatePod(ctx, pod)
				if errors.IsAlreadyExists(err) {
					logger.Warnf("The pod %s already exists.", pod.ObjectMeta.Name)
				} else if errors.IsTooManyRequests(err) || errors.IsTimeout(err) || errors.IsServerTimeout(err) {
					logger.Warnf("Fail to create pod %s with err: %v", pod.ObjectMeta.Name, err)
					// Retry to create pod due to timeout.
					scheduler.toCreatePods.PushFront(pod)
					time.Sleep(5 * time.Second)
				} else {
					logger.Warnf("Fail to create pod %s with err: %v", pod.ObjectMeta.Name, err)
					panic(err.Error())
				}
			}
		}
		time.Sleep(1 * time.Second)
	}
}

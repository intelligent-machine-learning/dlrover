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
	"time"

	"github.com/intelligent-machine-learning/dlrover/go/master/pkg/common"
	"github.com/intelligent-machine-learning/dlrover/go/master/pkg/kubeutils"
	logger "github.com/sirupsen/logrus"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
)

// ElasticScheduler launches pods without waiting for all resouces of pod are ready
type ElasticScheduler struct {
	k8sClient    *kubeutils.K8sClient
	toCreatePods *common.Queue
}

// NewElasticScheduler creates an elastic scheduler.
func NewElasticScheduler(k8sClient *kubeutils.K8sClient) *ElasticScheduler {
	return &ElasticScheduler{
		k8sClient:    k8sClient,
		toCreatePods: common.NewQueue(),
	}
}

// Start starts a routine to launch Pods.
func (scheduler *ElasticScheduler) Start(jobContext *common.JobContext) {
	go scheduler.createPodLoop(jobContext.NameSpace)
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
			pod := kubeutils.BuildPod(jobContext, podConfig)
			scheduler.toCreatePods.PushBack(pod)
		}
	}
}

func (scheduler *ElasticScheduler) createPodLoop(namespace string) {
	for {
		for scheduler.toCreatePods.Len() > 0 {
			pod := scheduler.toCreatePods.PopFront().(*corev1.Pod)
			err := scheduler.k8sClient.CreatePod(namespace, pod)
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
		time.Sleep(1 * time.Second)
	}
}

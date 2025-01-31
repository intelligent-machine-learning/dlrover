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
	"github.com/intelligent-machine-learning/dlrover/go/master/pkg/common"
	"github.com/intelligent-machine-learning/dlrover/go/master/pkg/kubernetes"
	logger "github.com/sirupsen/logrus"
)

// ElasticScheduler launches pods without waiting for all resouces of pod are ready
type ElasticScheduler struct {
	k8sClient *kubernetes.K8sClient
}

// DoScheduling creates/updates/deletes pods
func (scheduler *ElasticScheduler) DoScheduling(jobContext *common.JobContext, plan *SchedulingPlan) {
	for replicaType, spec := range plan.Replicas {
		for i := int32(0); i < spec.Replicas; i++ {
			replicaConfig := &kubernetes.ReplicaConfig{
				Type:   string(replicaType),
				ID:     i,
				Number: spec.Replicas,
				Rank:   i,
			}

			podConfig := &kubernetes.PodConfig{
				Replica:      replicaConfig,
				TemplateSpec: spec.Template.DeepCopy(),
			}
			pod := kubernetes.BuildPod(jobContext, podConfig)
			err := scheduler.k8sClient.CreatePod(jobContext.NameSpace, pod)
			scheduler.processAPIServerError(err)
		}
	}
}

func (scheduler *ElasticScheduler) processAPIServerError(err error) {
	if err != nil {
		logger.Infof("API server error : %v", err)
	}
}

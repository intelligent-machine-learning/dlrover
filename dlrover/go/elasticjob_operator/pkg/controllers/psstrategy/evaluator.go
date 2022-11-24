// Copyright 2022 The EasyDL Authors. All rights reserved.
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

package psstrategy

import (
	"context"
	"fmt"
	elasticv1alpha1 "github.com/intelligent-machine-learning/easydl/operator/api/v1alpha1"
	common "github.com/intelligent-machine-learning/easydl/operator/pkg/common"
	logger "github.com/sirupsen/logrus"
	runtime_client "sigs.k8s.io/controller-runtime/pkg/client"
)

const (
	evaluatorServicePort int = 2222
)

// EvaluatorManager generates an evaluator pod object.
type EvaluatorManager struct {
	PSTaskManager
}

func init() {
	logger.Infof("init evaluator manager")
	common.ReplicaManagers[ReplicaTypeEvaluator] = newEvaluatorManager()
}

func newEvaluatorManager() *EvaluatorManager {
	return &EvaluatorManager{
		PSTaskManager: PSTaskManager{
			taskType: ReplicaTypeEvaluator,
		},
	}
}

// ReconcilePods creates a Pod on a K8s cluster
func (m *EvaluatorManager) ReconcilePods(
	client runtime_client.Client,
	job *elasticv1alpha1.ElasticJob,
	resourceSpec *elasticv1alpha1.ReplicaResourceSpec,
) error {
	evaluatorStatus := m.getTaskStatus(job)
	aliveNum := int(evaluatorStatus.Active + evaluatorStatus.Pending)
	if aliveNum == 0 {
		evaluatorIndex := 0
		cluster := m.getPSCluster(client, job)
		if cluster.Evaluator == nil {
			cluster.Evaluator = make(map[int]string)
		}
		cluster.Evaluator[evaluatorIndex] = m.newTaskServiceAddr(
			job.Name, evaluatorIndex, evaluatorServicePort,
		)
		evaluator := m.newTask(job, evaluatorIndex)
		if evaluator == nil {
			return fmt.Errorf("No Evaluator ReplicaSpec")
		}
		m.insertTfConfigToEnv(&evaluator.Spec.Containers[0], cluster, evaluatorIndex)
		err := client.Create(context.Background(), evaluator)
		if err != nil {
			return err
		}
		service := m.newTaskService(job, evaluatorIndex, evaluatorServicePort)
		err = client.Create(context.Background(), service)
		if err != nil {
			return err
		}
	}
	return nil
}

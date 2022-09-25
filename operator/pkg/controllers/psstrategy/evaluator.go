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
	elasticv1alpha1 "github.com/intelligent-machine-learning/easydl/operator/api/v1alpha1"
	controllers "github.com/intelligent-machine-learning/easydl/operator/pkg/controllers"
	logger "github.com/sirupsen/logrus"
	corev1 "k8s.io/api/core/v1"
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
	controllers.ReplicaManagers[ReplicaTypeEvaluator] = newEvaluatorManager()
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
	r *controllers.ElasticJobReconciler,
	job *elasticv1alpha1.ElasticJob,
	resourceSpec *elasticv1alpha1.ReplicaResourceSpec,
) error {
	evaluatorStatus := m.getTaskStatus(job)
	aliveNum := int(evaluatorStatus.Active + evaluatorStatus.Pending)
	if aliveNum == 0 {
		evaluatorIndex := 0
		evaluator := m.newTask(job, evaluatorIndex)
		err := r.Create(context.Background(), evaluator)
		if err != nil {
			r.Recorder.Eventf(
				job,
				corev1.EventTypeWarning,
				string(corev1.PodFailed),
				"evaluator pod %s created failed: %v",
				evaluator.Name,
				err)
			return err
		}
		service := m.newTaskService(job, evaluatorIndex, evaluatorServicePort)
		err = r.Create(context.Background(), service)
		if err != nil {
			r.Recorder.Eventf(
				job,
				corev1.EventTypeWarning,
				string(corev1.PodFailed),
				"Evaluator service %s created failed: %v",
				service.Name,
				err,
			)
			return err
		}
	}
	return nil
}

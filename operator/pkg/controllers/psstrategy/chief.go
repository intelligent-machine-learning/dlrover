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
	chiefServicePort int = 2222
)

// ChiefManager generates a chief pod object.
type ChiefManager struct {
	PSTaskManager
}

func init() {
	logger.Infof("init chief manager")
	common.ReplicaManagers[ReplicaTypeChief] = newChiefManager()
}

func newChiefManager() *ChiefManager {
	return &ChiefManager{
		PSTaskManager: PSTaskManager{
			taskType: ReplicaTypeChief,
		},
	}
}

// ReconcilePods creates a Pod on a K8s cluster
func (m *ChiefManager) ReconcilePods(
	client runtime_client.Client,
	job *elasticv1alpha1.ElasticJob,
	resourceSpec *elasticv1alpha1.ReplicaResourceSpec,
) error {
	chiefStatus := m.getTaskStatus(job)
	aliveNum := int(chiefStatus.Active + chiefStatus.Pending)
	if aliveNum == 0 {
		chiefIndex := 0
		cluster := m.getPSCluster(client, job)
		if cluster.Chief == nil {
			cluster.Chief = make(map[int]string)
		}
		chief := m.newTask(job, chiefIndex)
		if chief == nil {
			return fmt.Errorf("No Chief ReplicaSpec")
		}
		m.insertTfConfigToEnv(&chief.Spec.Containers[0], cluster, chiefIndex)
		err := client.Create(context.Background(), chief)
		if err != nil {
			return err
		}
		service := m.newTaskService(job, chiefIndex, chiefServicePort)
		err = client.Create(context.Background(), service)
		if err != nil {
			return err
		}
	}
	return nil
}

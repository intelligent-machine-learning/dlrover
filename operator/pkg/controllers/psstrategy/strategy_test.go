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
	elasticv1alpha1 "github.com/intelligent-machine-learning/easydl/operator/api/v1alpha1"
	commonv1 "github.com/intelligent-machine-learning/easydl/operator/pkg/common/api/v1"
	"github.com/stretchr/testify/assert"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	runtime_client "sigs.k8s.io/controller-runtime/pkg/client"
	"testing"
)

func newTestJob() *elasticv1alpha1.ElasticJob {
	job := &elasticv1alpha1.ElasticJob{
		ObjectMeta: metav1.ObjectMeta{
			Name:        "test-psstrategy",
			Namespace:   "easydl",
			Annotations: map[string]string{},
			Labels:      map[string]string{},
		},
		Spec: elasticv1alpha1.ElasticJobSpec{
			ReplicaSpecs: map[commonv1.ReplicaType]*elasticv1alpha1.ReplicaSpec{},
		},
	}
	return job
}

func newTFcluster() ClusterSpec {
	cluster := ClusterSpec{}
	cluster[ReplicaTypeChief] = []string{"test-psstrategy-chief-0:2222"}
	cluster[ReplicaTypeEvaluator] = []string{"test-psstrategy-evaluator-0:2222"}
	workerHosts := []string{
		"test-psstrategy-worker-0:2222",
		"test-psstrategy-worker-1:2222",
		"test-psstrategy-worker-2:2222",
	}
	cluster[ReplicaTypeWorker] = workerHosts
	return cluster
}

func TestGetAllTaskHosts(t *testing.T) {
	job := newTestJob()
	manager := PSTaskManager{taskType: "worker"}
	taskStatus := &commonv1.ReplicaStatus{
		Active:  1,
		Pending: 2,
	}
	hosts := manager.getAllTaskHosts(job.Name, taskStatus, 2222)
	assert.Equal(
		t,
		hosts,
		[]string{
			"test-psstrategy-worker-0:2222",
			"test-psstrategy-worker-1:2222",
			"test-psstrategy-worker-2:2222",
		},
	)
}

func TestGetPSCluster(t *testing.T) {
	job := newTestJob()
	manager := PSTaskManager{taskType: ReplicaTypeWorker}
	job.Status.ReplicaStatuses = make(map[commonv1.ReplicaType]*commonv1.ReplicaStatus)
	job.Status.ReplicaStatuses[ReplicaTypeWorker] = &commonv1.ReplicaStatus{
		Active:  1,
		Pending: 2,
	}
	job.Status.ReplicaStatuses[ReplicaTypeChief] = &commonv1.ReplicaStatus{
		Active: 1,
	}
	job.Status.ReplicaStatuses[ReplicaTypeEvaluator] = &commonv1.ReplicaStatus{
		Active: 1,
	}
	client := runtime_client.NewDryRunClient(nil)
	cluster := manager.getPSCluster(client, job)
	expectedCluster := newTFcluster()
	assert.Equal(t, cluster, expectedCluster)
}

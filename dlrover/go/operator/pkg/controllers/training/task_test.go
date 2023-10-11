// Copyright 2022 The DLRover Authors. All rights reserved.
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

package training

import (
	"encoding/json"
	elasticv1alpha1 "github.com/intelligent-machine-learning/easydl/dlrover/go/operator/api/v1alpha1"
	common "github.com/intelligent-machine-learning/easydl/dlrover/go/operator/pkg/common"
	commonv1 "github.com/intelligent-machine-learning/easydl/dlrover/go/operator/pkg/common/api/v1"
	"github.com/stretchr/testify/assert"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	runtime_client "sigs.k8s.io/controller-runtime/pkg/client"
	"testing"
)

func newTestJob() *elasticv1alpha1.ElasticJob {
	job := &elasticv1alpha1.ElasticJob{
		ObjectMeta: metav1.ObjectMeta{
			Name:        "test-training",
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

func TestGetAllTaskHosts(t *testing.T) {
	job := newTestJob()
	manager := TaskManager{taskType: "worker"}
	hosts := manager.getAllTaskHosts(job.Name, 3, 2222)
	assert.Equal(
		t,
		hosts,
		[]string{
			"test-training-edljob-worker-0:2222",
			"test-training-edljob-worker-1:2222",
			"test-training-edljob-worker-2:2222",
		},
	)
}

func TestGetPSCluster(t *testing.T) {
	job := newTestJob()
	manager := TaskManager{taskType: ReplicaTypeChief}
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
	job.Status.ReplicaStatuses[ReplicaTypePS] = &commonv1.ReplicaStatus{
		Initial: 1,
	}
	podMeta := &elasticv1alpha1.PodMeta{
		Name:      "chief-0",
		ID:        0,
		RankIndex: 0,
		Type:      ReplicaTypeChief,
		Service:   "test-training-chief-0:2222",
		Resource: corev1.ResourceList{
			corev1.ResourceCPU:              resource.MustParse("1"),
			corev1.ResourceMemory:           resource.MustParse("1Gi"),
			corev1.ResourceEphemeralStorage: resource.MustParse("1Gi"),
		},
	}
	scalePlan := &elasticv1alpha1.ScalePlan{
		Spec: elasticv1alpha1.ScaleSpec{
			PsHosts: []string{"test-training-ps-0:3333"},
		},
	}
	client := runtime_client.NewDryRunClient(nil)
	cluster := manager.getPSClusterForPod(client, job, scalePlan, podMeta)
	expectedCluster := SparseClusterSpec{
		PS:    []string{"test-training-ps-0:3333"},
		Chief: map[int]string{0: "test-training-chief-0:2222"},
	}
	assert.Equal(t, cluster, expectedCluster)
}

func TestNewTaskPod(t *testing.T) {
	job := newTestJob()
	container := corev1.Container{
		Name:            "main",
		Image:           "test",
		ImagePullPolicy: corev1.PullAlways,
		Command:         []string{"/bin/bash", "echo 0"},
	}

	job.Spec.ReplicaSpecs[ReplicaTypeChief] = &elasticv1alpha1.ReplicaSpec{
		ReplicaSpec: commonv1.ReplicaSpec{
			Template: corev1.PodTemplateSpec{
				Spec: corev1.PodSpec{
					Containers:    []corev1.Container{container},
					RestartPolicy: corev1.RestartPolicyNever,
				},
			},
		},
		RestartCount: 3,
	}
	podMeta := &elasticv1alpha1.PodMeta{
		Name:      "test-training-edljob-chief-0",
		ID:        0,
		RankIndex: 0,
		Type:      ReplicaTypeChief,
		Service:   "test-training-edljob-chief-0:2222",
		Resource: corev1.ResourceList{
			corev1.ResourceCPU:              resource.MustParse("1"),
			corev1.ResourceMemory:           resource.MustParse("1Gi"),
			corev1.ResourceEphemeralStorage: resource.MustParse("1Gi"),
		},
	}

	manager := &TaskManager{taskType: ReplicaTypeChief}
	pod := manager.newTask(job, podMeta)
	assert.Equal(t, pod.Name, "test-training-edljob-chief-0")
	assert.Equal(t, pod.Labels[LabelRestartCount], "3")
	assert.Equal(
		t,
		pod.Labels[common.LabelReplicaTypeKey],
		string(ReplicaTypeChief),
	)
	assert.Equal(
		t,
		pod.Spec.Containers[0].Resources.Requests.Cpu().AsApproximateFloat64(),
		float64(1),
	)
	assert.Equal(t, len(pod.Spec.Containers[0].Env), 6)
	assert.Equal(t, pod.Spec.Containers[0].Env[0].Name, "DLROVER_MASTER_ADDR")
	assert.Equal(
		t,
		pod.Spec.Containers[0].Env[0].Value,
		"elasticjob-test-training-dlrover-master:50001",
	)

	cluster := SparseClusterSpec{
		Chief:  map[int]string{0: "test-training-chief-0:2222"},
		Worker: map[int]string{0: "test-training-worker-0:2222", 1: "test-training-worker-1:2222"},
		PS:     []string{"test-training-ps-0:3333"},
	}

	clusterSpec := convertSparseClusterToCluster(cluster)
	assert.Equal(t, clusterSpec.Worker[0], "test-training-worker-0:2222")
	assert.Equal(t, clusterSpec.Worker[1], "test-training-worker-1:2222")

	InsertTfConfigToEnv(&container, cluster, ReplicaTypeChief, 0)
	tfConfig := SparseTFConfig{}
	err := json.Unmarshal([]byte(container.Env[0].Value), &tfConfig)
	assert.NoError(t, err)
	assert.Equal(t, tfConfig.Task.Type, ReplicaTypeChief)
	assert.Equal(t, tfConfig.Task.Index, 0)
}

func TestNewPodService(t *testing.T) {
	job := newTestJob()
	podMeta := &elasticv1alpha1.PodMeta{
		Name:      "test-training-edljob-chief-0",
		ID:        0,
		RankIndex: 0,
		Type:      ReplicaTypeChief,
		Service:   "test-training-edljob-chief-0:2222",
		Resource: corev1.ResourceList{
			corev1.ResourceCPU:              resource.MustParse("1"),
			corev1.ResourceMemory:           resource.MustParse("1Gi"),
			corev1.ResourceEphemeralStorage: resource.MustParse("1Gi"),
		},
	}
	chiefManager := &TaskManager{taskType: ReplicaTypeChief}
	chiefService := chiefManager.newServiceForPod(job, podMeta)
	assert.Equal(
		t,
		chiefService.Spec.Selector[common.LabelReplicaTypeKey],
		string(ReplicaTypeChief),
	)
	assert.Equal(t, chiefService.Spec.Selector[common.LabelRankIndexKey], "0")

	podMeta.Name = "test-training-edljob-worker-0"
	podMeta.Type = ReplicaTypeWorker
	podMeta.Service = "test-training-edljob-worker-0:2222"

	workerManager := &TaskManager{taskType: ReplicaTypeWorker}
	workerService := workerManager.newServiceForPod(job, podMeta)
	assert.Equal(
		t,
		workerService.Spec.Selector[common.LabelReplicaTypeKey],
		string(ReplicaTypeWorker),
	)
	assert.Equal(t, workerService.Spec.Selector[common.LabelRankIndexKey], "0")
}

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

package kubeutils

import (
	"fmt"

	elasticjobv1 "github.com/intelligent-machine-learning/dlrover/go/elasticjob/api/v1alpha1"
	"github.com/intelligent-machine-learning/dlrover/go/master/pkg/common"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

const (
	envMasterAddr  = "DLROVER_MASTER_ADDR"
	envPodName     = "MY_POD_NAME"
	envPodIP       = "MY_POD_IP"
	envHostIP      = "MY_HOST_IP"
	envReplicaType = "REPLICA_TYPE"
	envReplicaID   = "REPLICA_ID"
	envReplicaRank = "REPLICA_RANK"
	envReplicaNum  = "REPLICA_NUM"

	labelJobKey         = "elasticjob.dlrover/name"
	labelReplicaTypeKey = "elasticjob.dlrover/replica-type"
	labelReplicaIDKey   = "elasticjob.dlrover/replica-id"
	labelReplicaRankKey = "elasticjob.dlrover/rank"
)

// ReplicaConfig contains the replica specification.
type ReplicaConfig struct {
	Type string
	ID   int32
	// Number if the total number of the replicas.
	Number int32
	// Rank is the rank of the pod in the replicas.
	Rank int32
}

// PodConfig contains the replica config and pod template spec.
type PodConfig struct {
	Replica      *ReplicaConfig
	TemplateSpec *corev1.PodTemplateSpec
}

// BuildPod builds a corev1.Pod.
func BuildPod(jobContext *common.JobContext, podConfig *PodConfig, ownerJob *elasticjobv1.ElasticJob) *corev1.Pod {
	podName := fmt.Sprintf("%s-%s-%d", jobContext.Name, podConfig.Replica.Type, podConfig.Replica.ID)
	pod := &corev1.Pod{
		TypeMeta: metav1.TypeMeta{
			APIVersion: "v1",
			Kind:       "Pod",
		},
		ObjectMeta: podConfig.TemplateSpec.ObjectMeta,
		Spec:       podConfig.TemplateSpec.Spec,
	}
	// Set pod name and namespace.
	pod.ObjectMeta.Name = podName
	pod.ObjectMeta.Namespace = jobContext.NameSpace
	pod.ObjectMeta.OwnerReferences = []metav1.OwnerReference{
		*metav1.NewControllerRef(ownerJob, elasticjobv1.SchemeGroupVersionKind),
	}

	if pod.ObjectMeta.Labels == nil {
		pod.ObjectMeta.Labels = make(map[string]string)
	}

	// Insert Replica specifications into the pod labels.
	pod.ObjectMeta.Labels[labelJobKey] = jobContext.Name
	pod.ObjectMeta.Labels[labelReplicaTypeKey] = podConfig.Replica.Type
	pod.ObjectMeta.Labels[labelReplicaIDKey] = fmt.Sprintf("%d", podConfig.Replica.ID)
	pod.ObjectMeta.Labels[labelReplicaRankKey] = fmt.Sprintf("%d", podConfig.Replica.Rank)

	mainContainer := &pod.Spec.Containers[0]
	insertJobMasterAddrEnv(mainContainer, jobContext.MasterHost, jobContext.MasterPort)
	insertPodMetaEnv(mainContainer)
	insertReplicaEnv(mainContainer, podConfig.Replica)

	return pod
}

func insertJobMasterAddrEnv(container *corev1.Container, host string, port int) {
	jobMasterServiceEnv := corev1.EnvVar{
		Name:  envMasterAddr,
		Value: fmt.Sprintf("%s:%d", host, port),
	}
	container.Env = append(container.Env, jobMasterServiceEnv)

}

func insertPodMetaEnv(container *corev1.Container) {
	podNameEnv := corev1.EnvVar{
		Name: envPodName,
		ValueFrom: &corev1.EnvVarSource{
			FieldRef: &corev1.ObjectFieldSelector{
				APIVersion: "v1",
				FieldPath:  "metadata.name",
			},
		},
	}
	container.Env = append(container.Env, podNameEnv)

	podIPEnv := corev1.EnvVar{
		Name: envPodIP,
		ValueFrom: &corev1.EnvVarSource{
			FieldRef: &corev1.ObjectFieldSelector{
				APIVersion: "v1",
				FieldPath:  "status.podIP",
			},
		},
	}
	container.Env = append(container.Env, podIPEnv)

	hostIPEnv := corev1.EnvVar{
		Name: envHostIP,
		ValueFrom: &corev1.EnvVarSource{
			FieldRef: &corev1.ObjectFieldSelector{
				APIVersion: "v1",
				FieldPath:  "status.hostIP",
			},
		},
	}
	container.Env = append(container.Env, hostIPEnv)
}

func insertReplicaEnv(container *corev1.Container, replicaConfig *ReplicaConfig) {
	replicaTypeEnv := corev1.EnvVar{
		Name:  envReplicaType,
		Value: string(replicaConfig.Type),
	}
	container.Env = append(container.Env, replicaTypeEnv)

	replicaIDEnv := corev1.EnvVar{
		Name:  envReplicaID,
		Value: fmt.Sprintf("%d", replicaConfig.ID),
	}
	container.Env = append(container.Env, replicaIDEnv)

	rankIDEnv := corev1.EnvVar{
		Name:  envReplicaNum,
		Value: fmt.Sprintf("%d", replicaConfig.Rank),
	}
	container.Env = append(container.Env, rankIDEnv)

	replicaNumEnv := corev1.EnvVar{
		Name:  envReplicaRank,
		Value: fmt.Sprintf("%d", replicaConfig.Number),
	}
	container.Env = append(container.Env, replicaNumEnv)
}

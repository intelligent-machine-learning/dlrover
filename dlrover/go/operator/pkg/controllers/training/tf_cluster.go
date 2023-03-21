// Copyright 2023 The DLRover Authors. All rights reserved.
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
	commonv1 "github.com/intelligent-machine-learning/easydl/dlrover/go/operator/pkg/common/api/v1"
	logger "github.com/sirupsen/logrus"
	corev1 "k8s.io/api/core/v1"
)

// TaskSpec is the specification for a task (PS or worker) of the ElasticJob using
// ParameterServerStrategy.
type TaskSpec struct {
	Type  commonv1.ReplicaType `json:"type"`
	Index int                  `json:"index"`
}

// SparseClusterSpec enables a server to be configured without needing to know
// the identity of (for example) all other worker tasks.
// https://www.tensorflow.org/api_docs/python/tf/train/ClusterSpec
type SparseClusterSpec struct {
	Worker    map[int]string `json:"worker,omitempty"`
	PS        []string       `json:"ps"`
	Chief     map[int]string `json:"chief"`
	Evaluator map[int]string `json:"evaluator,omitempty"`
}

// SparseTFConfig is a struct representing the distributed TensorFlow config.
type SparseTFConfig struct {
	// Cluster represents a TensorFlow ClusterSpec.
	// See: https://www.tensorflow.org/api_docs/python/tf/train/ClusterSpec
	Cluster SparseClusterSpec `json:"sparseCluster"`
	Task    TaskSpec          `json:"task"`
}

// ClusterSpec represents a cluster TensorFlow specification.
// https://www.tensorflow.org/deploy/distributed#create_a_tftrainclusterspec_to_describe_the_cluster
// It is a map from job names to network addresses.
type ClusterSpec struct {
	Worker    []string `json:"worker,omitempty"`
	PS        []string `json:"ps"`
	Chief     []string `json:"chief"`
	Evaluator []string `json:"evaluator,omitempty"`
}

// TFConfig is a struct representing the distributed TensorFlow config.
type TFConfig struct {
	// Cluster represents a TensorFlow ClusterSpec.
	// See: https://www.tensorflow.org/api_docs/python/tf/train/ClusterSpec
	Cluster ClusterSpec `json:"cluster"`
	Task    TaskSpec    `json:"task"`
}

// InsertTfConfigToEnv inserts TFConfig to envs
func InsertTfConfigToEnv(
	container *corev1.Container,
	cluster SparseClusterSpec,
	taskType commonv1.ReplicaType,
	rankIndex int,
) {
	clusterSpec := convertSparseClusterToCluster(cluster)
	tfConfig := TFConfig{
		Cluster: *clusterSpec,
		Task: TaskSpec{
			Type:  taskType,
			Index: rankIndex,
		},
	}
	tfConfigJSONByteSlice, err := json.Marshal(tfConfig)
	tfConfigStr := string(tfConfigJSONByteSlice)
	tfConfigEnv := corev1.EnvVar{
		Name:  EnvTfConfigName,
		Value: tfConfigStr,
	}
	if err != nil {
		logger.Infof("Failed to get TFCONFIG %v", err)
		return
	}
	container.Env = append(container.Env, tfConfigEnv)
}

func convertSparseClusterToCluster(sparseCluster SparseClusterSpec) *ClusterSpec {
	clusterSpec := &ClusterSpec{}
	for i := 0; i < len(sparseCluster.Worker); i++ {
		clusterSpec.Worker = append(clusterSpec.Worker, sparseCluster.Worker[i])
	}
	for _, host := range sparseCluster.Chief {
		clusterSpec.Chief = append(clusterSpec.Chief, host)
	}
	for _, host := range sparseCluster.Evaluator {
		clusterSpec.Evaluator = append(clusterSpec.Evaluator, host)
	}
	clusterSpec.PS = append(clusterSpec.PS, sparseCluster.PS...)
	return clusterSpec
}

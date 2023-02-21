/*
Copyright 2022.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package v1alpha1

import (
	commonv1 "github.com/intelligent-machine-learning/easydl/dlrover/go/operator/pkg/common/api/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// EDIT THIS FILE!  THIS IS SCAFFOLDING FOR YOU TO OWN!
// NOTE: json tags are required.  Any new fields you add must have json tags for the fields to be serialized.

// ElasticJobSpec defines the desired state of ElasticJob
type ElasticJobSpec struct {
	// INSERT ADDITIONAL SPEC FIELDS - desired state of cluster
	// Important: Run "make" to regenerate code after modifying this file

	// DistributionStrategy specifies the distribution strategy of a job.
	// Now, the strategy supports parameter-server and ring-allreduce.
	DistributionStrategy string `json:"distributionStrategy,omitempty"`

	// ResourceLimits specifies the maximum resource of a job. For example,
	// {"cpu": "100", "memory": "10240Mi"} means the number of maximum cpu cores
	// is 100 and the maximum of all Pods is 10Gi.
	ResourceLimits map[string]string `json:"resourceLimits,omitempty"`

	// optimizeMode specifies the mode to optimize the job resoruce.
	// Now, it support "mannual/single-job/cluster".
	OptimizeMode string `json:"optimizeMode,omitempty"`

	// BrainService specifies the address of Brain to optimize the job resource.
	// It is only used when optimizeMode is cluster.
	BrainService string `json:"brainService,omitempty"`

	// EnableElasticScheduling starts elasticity of Pods.
	EnableElasticScheduling bool `json:"enableElasticScheduling,omitempty"`

	// EnableDynamicSharding starts the dynamic sharding of the dataset.
	EnableDynamicSharding bool `json:"enableDynamicSharding,omitempty"`

	// A map of ReplicaType (type) to ReplicaSpec (value). Specifies the training cluster configuration.
	// For example,
	//   {
	//     "PS": ReplicaSpec,
	//     "Worker": ReplicaSpec,
	//   }
	ReplicaSpecs map[commonv1.ReplicaType]*ReplicaSpec `json:"replicaSpecs"`

	// Envs specifies environment variables for Pods of the job.
	Envs map[string]*corev1.EnvVar `json:"envs,omitempty"`
}

// ReplicaSpec specifies the number and resources of replica.
type ReplicaSpec struct {
	commonv1.ReplicaSpec `json:",inline"`

	// RestartCount is the number of relaunching a failed replica.
	RestartCount int `json:"restartCount,omitempty"`

	// AutoScale is the flag to auto scale the number of replicas and resource of each replica.
	AutoScale bool `json:"autoScale,omitempty"`

	// RestartTimeout is the time to wait a hanging replica.
	RestartTimeout int `json:"restartTimeout,omitempty"`

	// Priority supports high/low/0.5. The 0.5 means that half workers have high priority,
	// and half workers have low priority. The default value is low.
	Priority string `json:"priority,omitempty"`
}

// ElasticJobStatus defines the observed state of ElasticJob
type ElasticJobStatus struct {
	// INSERT ADDITIONAL STATUS FIELD - define observed state of cluster
	// Important: Run "make" to regenerate code after modifying this file

	commonv1.JobStatus `json:"status,omitempty"`

	// Phase shows the phase of job lifecycle
	Phase commonv1.JobConditionType `json:"phase,omitempty"`

	// ScalePlan is a name of ScalePlan CRD to scale the job resource.
	ScalePlan string `json:"scalePlan,omitempty"`
}

//+kubebuilder:object:root=true
//+kubebuilder:subresource:status
// +resource:path=elasticjob
// +kubebuilder:resource:scope=Namespaced
// +kubebuilder:printcolumn:name="Phase",type=string,JSONPath=`.status.phase`
// +kubebuilder:printcolumn:name="Age",type=date,JSONPath=`.metadata.creationTimestamp`

// ElasticJob is the Schema for the elasticjobs API
type ElasticJob struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   ElasticJobSpec   `json:"spec,omitempty"`
	Status ElasticJobStatus `json:"status,omitempty"`
}

//+kubebuilder:object:root=true

// ElasticJobList contains a list of ElasticJob
type ElasticJobList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []ElasticJob `json:"items"`
}

func init() {
	SchemeBuilder.Register(&ElasticJob{}, &ElasticJobList{})
}

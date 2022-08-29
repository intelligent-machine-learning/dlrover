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
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// EDIT THIS FILE!  THIS IS SCAFFOLDING FOR YOU TO OWN!
// NOTE: json tags are required.  Any new fields you add must have json tags for the fields to be serialized.

// ElasticJobSpec defines the desired state of ElasticJob
type ElasticJobSpec struct {
	// INSERT ADDITIONAL SPEC FIELDS - desired state of cluster
	// Important: Run "make" to regenerate code after modifying this file

	// Command is the entrypoint for Pods of the job.
	Command string `json:"command,omitempty"`

	//Image is the image for Pods of the job.
	Image string `json:"Image,omitempty"`

	// ParameterServer specifies the resources of PS for the job.
	ParameterServer *ReplicaSpec `json:"parameter_server,omitempty"`

	// Worker specifies the resources of workers for the job.
	Worker *ReplicaSpec `json:"worker,omitempty"`

	// Evaluator specifies the resource of evaluators for the job.
	Evaluator *ReplicaSpec `json:"evaluator,omitempty"`

	// Envs specifies environment variables for Pods of the job.
	Envs map[string]*corev1.EnvVar `json:"envs,omitempty"`
}

type ReplicaSpec struct {
	// Count is the requested number of replicas
	Count int `json:"count,omitempty"`
	// Resource is the requested resource of a replica
	Resource * ResourceSpec `json:"resource,omitempty"`
}

type ResourceSpec struct {
	// CPU is the requested CPU cores of a replica
	CPU int32 `json:"cpu,omitempty"`
	// Memory is the requested memory (MB) of a replica
	Memory int32 `json:"memory,omitempty"`
	// GPU is the requested GPU of a replica
	GPU string `json:"gpu,omitempty"`
}

// ElasticJobStatus defines the observed state of ElasticJob
type ElasticJobStatus struct {
	// INSERT ADDITIONAL STATUS FIELD - define observed state of cluster
	// Important: Run "make" to regenerate code after modifying this file
}

//+kubebuilder:object:root=true
//+kubebuilder:subresource:status

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

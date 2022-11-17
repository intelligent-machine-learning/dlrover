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
	commonv1 "github.com/intelligent-machine-learning/easydl/operator/pkg/common/api/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// EDIT THIS FILE!  THIS IS SCAFFOLDING FOR YOU TO OWN!
// NOTE: json tags are required.  Any new fields you add must have json tags for the fields to be serialized.

// ScalerSpec defines the desired resource state of an ElasticJob
type ScalerSpec struct {
	// INSERT ADDITIONAL SPEC FIELDS - desired state of cluster
	// Important: Run "make" to regenerate code after modifying this file

	// A map of ReplicaType (type) to ReplicaSpec (value). Specifies the resource of a job.
	// For example,
	//   {
	//     "PS": ReplicaResourceSpec,
	//     "worker": ReplicaResourceSpec,
	//   }
	ReplicaResourceSpecs map[commonv1.ReplicaType]ReplicaResourceSpec `json:"replicaResourceSpec,omitempty"`

	// NodeResourceSpec specifics resources configurations of nodes.
	NodeResourceSpec map[string]ResourceSpec `json:"nodeResourceSpec,omitempty"`

	// OwnerJob specifies a job to scale.
	OwnerJob string `json:"ownerJob,omitempty"`
}

// ReplicaResourceSpec specifies the number and resources of replica.
type ReplicaResourceSpec struct {
	// Replicas is the number of replica
	Replicas int `json:"replicas,omitempty"`

	// Resource defines the resource of each replica
	Resource ResourceSpec `json:"resource,omitempty"`
}

// ResourceSpec specifies the resources of a repalica
type ResourceSpec struct {
	// CPU is the requested CPU cores of a replica
	CPU string `json:"cpu,omitempty"`
	// Memory is the requested memory (MB) of a replica
	Memory string `json:"memory,omitempty"`
	// GPU is the requested GPU of a replica
	GPU string `json:"gpu,omitempty"`
}

// ScalerStatus defines the observed state of ElasticJob
type ScalerStatus struct {
	// INSERT ADDITIONAL STATUS FIELD - define observed state of cluster
	// Important: Run "make" to regenerate code after modifying this file

	// CreateTime represents time when the scaler was acknowledged by the scaler controller.
	CreateTime *metav1.Time `json:"createTime,omitempty"`
}

//+kubebuilder:object:root=true
//+kubebuilder:subresource:status
// +resource:path=scaler
// +kubebuilder:resource:scope=Namespaced
// +kubebuilder:printcolumn:name="Age",type=date,JSONPath=`.metadata.creationTimestamp`

// Scaler is the Schema for the scalers API
type Scaler struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   ScalerSpec   `json:"spec,omitempty"`
	Status ScalerStatus `json:"status,omitempty"`
}

//+kubebuilder:object:root=true

// ScalerList contains a list of Scale
type ScalerList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []Scaler `json:"items"`
}

func init() {
	SchemeBuilder.Register(&Scaler{}, &ScalerList{})
}

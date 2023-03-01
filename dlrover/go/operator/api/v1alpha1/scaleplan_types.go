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

// ScaleSpec defines the desired resource state of an ElasticJob
type ScaleSpec struct {
	// INSERT ADDITIONAL SPEC FIELDS - desired state of cluster
	// Important: Run "make" to regenerate code after modifying this file

	// A map of ReplicaType (type) to ReplicaSpec (value). Specifies the resource of a job.
	// For example,
	//   {
	//     "PS": ReplicaResourceSpec,
	//     "worker": ReplicaResourceSpec,
	//   }
	ReplicaResourceSpecs map[commonv1.ReplicaType]ReplicaResourceSpec `json:"replicaResourceSpecs,omitempty"`

	// CreatePods are Pods to be created.
	CreatePods []PodMeta `json:"createPods,omitempty"`

	// RemovePods are Pods to be removed
	RemovePods []PodMeta `json:"removePods,omitempty"`

	// MigratePods are Pods to be migrated to other Pods with the new resource.
	MigratePods []PodMeta `json:"migratePods,omitempty"`

	// PsHosts are hosts of PS Pods
	PsHosts []string `json:"psHosts,omitempty"`

	// OwnerJob specifies a job to scale.
	OwnerJob string `json:"ownerJob,omitempty"`
}

// ReplicaResourceSpec specifies the number and resources of replica.
type ReplicaResourceSpec struct {
	// Replicas is the number of replica
	Replicas int `json:"replicas,omitempty"`

	// Resource defines the resource of each replica
	Resource corev1.ResourceList `json:"resource,omitempty"`
}

// PodMeta specifies the meta of a Pod.
type PodMeta struct {
	// Name is the name of the Pod
	Name string `json:"name,omitempty"`

	// Id is the identity of the Pod
	ID int `json:"id,omitempty"`

	// Type is the type of the Pod
	Type commonv1.ReplicaType `json:"type,omitempty"`

	// RankIndex is the index of the Pod
	RankIndex int `json:"rankIndex,omitempty"`

	// Service is the service whose endpoint is the Pod.
	Service string `json:"service,omitempty"`

	// Resource defines the resource of each replica
	Resource corev1.ResourceList `json:"resource,omitempty"`
}

// ScalePlanStatus defines the observed state of ElasticJob
type ScalePlanStatus struct {
	// INSERT ADDITIONAL STATUS FIELD - define observed state of cluster
	// Important: Run "make" to regenerate code after modifying this file

	// CreateTime represents time when the scaling plan was acknowledged by the controller.
	CreateTime *metav1.Time `json:"createTime,omitempty"`

	// FinishTime represents time when the scaling plan is executed by the controller.
	FinishTime *metav1.Time `json:"finishTime,omitempty"`

	// Phase shows the phase of scalePlan lifecycle
	Phase commonv1.JobConditionType `json:"phase,omitempty"`
}

//+kubebuilder:object:root=true
//+kubebuilder:subresource:status
// +resource:path=scaleplan
// +kubebuilder:resource:scope=Namespaced
// +kubebuilder:printcolumn:name="Phase",type=string,JSONPath=`.status.phase`
// +kubebuilder:printcolumn:name="Age",type=date,JSONPath=`.metadata.creationTimestamp`

// ScalePlan is the Schema for the scaling plan API
type ScalePlan struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   ScaleSpec       `json:"spec,omitempty"`
	Status ScalePlanStatus `json:"status,omitempty"`
}

//+kubebuilder:object:root=true

// ScalePlanList contains a list of ScalePlan
type ScalePlanList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []ScalePlan `json:"items"`
}

func init() {
	SchemeBuilder.Register(&ScalePlan{}, &ScalePlanList{})
}

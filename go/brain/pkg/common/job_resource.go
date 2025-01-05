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

package common

// PodResource includes a pod's resource information
type PodResource struct {
	// CPUCore is the used cpu cores
	CPUCore float32
	// Memory is the memory size
	Memory float64
	// GPUCore is the used gpu cores
	GPUCore float32
	// GPUType is the gpu type
	GPUType string
}

// TaskGroupResource is the resource of a task group
type TaskGroupResource struct {
	// Count is the number of replica
	Count int32
	// Resource is the resource of pods
	Resource *PodResource
}

// JobResource is the resource of a job
type JobResource struct {
	// TaskGroupResources is the resources of task groups
	TaskGroupResources map[string]*TaskGroupResource
	// PodResources is the resources of particular pods
	PodResources map[string]*PodResource
}

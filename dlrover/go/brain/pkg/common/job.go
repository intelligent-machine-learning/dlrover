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

import "time"

const (
	// WorkerTaskGroupName is the name of worker task group
	WorkerTaskGroupName = "worker"
	// PSTaskGroupName is the name of ps task group
	PSTaskGroupName = "ps"
)

// JobMeta is the struct of job meta
type JobMeta struct {
	Name      string
	UUID      string
	User      string
	Cluster   string
	Namespace string
	State     *JobState
}

// PodState is the struct of the pod state
type PodState struct {
	Name           string
	UUID           string
	Type           string
	IsOOM          bool
	CustomizedData map[string]string
}

// JobState is the struct of the job state
type JobState struct {
	PodStates      map[string]*PodState
	CustomizedData map[string]string
}

// JobMetrics is the struct of job metrics
type JobMetrics struct {
	JobUUID            string
	JobName            string
	CreatedAt          time.Time
	FinishedAt         time.Time
	HyperParamsFeature string
	JobFeature         string
	DatasetFeature     string
	ModelFeature       string
	JobRuntime         string
	ExitReason         string
	Optimization       string
	Type               string
	Resource           string
	CustomizedData     string
}

// OptimizeJobMeta includes required information for optimization
type OptimizeJobMeta struct {
	JobMeta        *JobMeta
	Metrics        *JobMetrics
	CustomizedData map[string]string
}

// JobNodeStatus is the struct of job node status
type JobNodeStatus struct {
	IsOOM  bool
	Status string
}

// JobStatus is the struct of job stauts
type JobStatus struct {
	IsOOM bool
}

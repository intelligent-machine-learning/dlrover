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

const (
	// DefaultQueryBackwardTimePeriodInHour is the default hours backward when to search jobs
	DefaultQueryBackwardTimePeriodInHour = 7 * 24
	// ExitReasonCompleted is the completed exit reason
	ExitReasonCompleted = "Completed"
	// DefaultMemoryMarginPercent is the default memory margin percent
	DefaultMemoryMarginPercent = 0.5
	// DefaultCPUMargin is the default cpu margin
	DefaultCPUMargin = 4
	// DefaultWorkerCreateCPU is the default value of CPU for first worker.
	DefaultWorkerCreateCPU = 12
	// DefaultMaxPSCount if the max count of PS
	DefaultMaxPSCount = 15
	// DefaultPSMemoryMarginPercent is the default PS memory margin percent
	DefaultPSMemoryMarginPercent = 0.2
	// DefaultPSCPUMarginPercent is the default PS memory margin percent
	DefaultPSCPUMarginPercent = 0.5
	// DefaultWorkerMaxCPU is the maximum cpu for each worker
	DefaultWorkerMaxCPU = 32
	// DefaultWorkerMaxMemory is the maximum memory for each worker
	DefaultWorkerMaxMemory = 32000000000
	// DefaultPSMinCPU is the default minimum cpu for each ps
	DefaultPSMinCPU = 12
	// DefaultPSMinReplica is the default minimum ps replica in a job
	DefaultPSMinReplica = 1
	// DefaultWorkerMinCPU is the default minimum cpu for each worker
	DefaultWorkerMinCPU = 8
	// DefaultWorkerMinReplica is the default minimum worker replica in a job
	DefaultWorkerMinReplica = 2
	// DefaultMaxPSMemory is the max memory of a PS.
	DefaultMaxPSMemory = 68719476736
	//NRecordToAvgResource is the number of record to compute the average resource usage.
	NRecordToAvgResource = 3
)

const (
	// ResourceTypePSCPU is the job node resource type of ps cpu
	ResourceTypePSCPU = "ps.cpu"
	// ResourceTypePSMemory is the job node resource type of ps memory
	ResourceTypePSMemory = "ps.mem"
	// ResourceTypeWorkerCPU is the job node resource type of worker cpu
	ResourceTypeWorkerCPU = "worker.cpu"
	// ResourceTypeWorkerMemory is the job node resource type of worker memory
	ResourceTypeWorkerMemory = "worker.memory"

	// ResourceTypeCPU is cpu resource type
	ResourceTypeCPU = "cpu"
	// ResourceTypeMemory is memory resource type
	ResourceTypeMemory = "memory"
)

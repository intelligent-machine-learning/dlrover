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

package config

const (
	// Namespace is the jobmanagement key of namespace
	Namespace = "namespace"
	// KubeClientInterface is the jobmanagement key of kube client
	KubeClientInterface = "kube.client.interface"
	// Cluster is the jobmanagement key of cluster
	Cluster = "cluster"

	// DBUser is the jobmanagement key of database user
	DBUser = "db.user"
	// DBPassword is the jobmanagement key of database password
	DBPassword = "db.password"
	// DBEngineType is the jobmanagement key of database engine type, e.g., mysql
	DBEngineType = "db.engine.type"
	// DBURL is the jobmanagement key of database url
	DBURL = "db.url"

	// DataStoreConfigMapName is the name of data store jobmanagement map
	DataStoreConfigMapName = "data-store.jobmanagement-map.name"
	// DataStoreConfigMapKey is the key of data store jobmanagement map
	DataStoreConfigMapKey = "data-store.jobmanagement-map.key"
	// DataStoreName is the name of a data store
	DataStoreName = "data-store.name"

	// OptimizerConfigMapName is the name of optimization jobmanagement map
	OptimizerConfigMapName = "optimization.jobmanagement-map.name"
	// OptimizerConfigMapKey is the key of optimization jobmanagement map
	OptimizerConfigMapKey = "optimization.jobmanagement-map.key"

	// KubeWatcherConfigMapName is the name of kube watcher jobmanagement map
	KubeWatcherConfigMapName = "kube-watcher.jobmanagement-map.name"
	// KubeWatcherConfigMapKey is the key of kube watcher jobmanagement map
	KubeWatcherConfigMapKey = "kube-watcher.jobmanagement-map.key"
	// KubeWatcherMetricsAddress is the address of kube watcher metrics
	KubeWatcherMetricsAddress = "kube-watcher.metrics-address"
	// KubeWatcherEnableLeaderElect is key of if enable leader select of kube watcher
	KubeWatcherEnableLeaderElect = "kube-watcher.leader-elect.enable"
	// KubeWatcherHandlerConfigMapName is the name of kube watcher handler jobmanagement map
	KubeWatcherHandlerConfigMapName = "kube-watcher.handler.jobmanagement-map.name"
	// KubeWatcherHandlerConfigMapKey is the key of kube watcher handler jobmanagement map
	KubeWatcherHandlerConfigMapKey = "kube-watcher.handler.jobmanagement-map.key"

	// BrainServerConfigMapName is the name of brain server jobmanagement map
	BrainServerConfigMapName = "brain.server.jobmanagement-map.name"
	// BrainServerConfigMapKey is the name of brain server jobmanagement key
	BrainServerConfigMapKey = "brain.server.jobmanagement-map.key"

	// QueryBackwardTimePeriodInHour is the jobmanagement key for query backward time period in hour
	QueryBackwardTimePeriodInHour = "query.backward.time-period.hour"

	// JobNodeMemoryMarginPercent is the jobmanagement key for job node memory margin percent
	JobNodeMemoryMarginPercent = "job.node.memory.margin.percent"
	// JobNodeCPUMargin is the jobmanagement key for job node cpu margin
	JobNodeCPUMargin = "job-node.cpu.margin"

	// OptimizerMinWorkerCreateCPU is the key of min CPU of the first worker
	OptimizerMinWorkerCreateCPU = "optimization.worker.create.cpu.min"
	// OptimizerWorkerOomMemoryMarginPercent is the jobmanagement key of oom worker memory margin percent
	OptimizerWorkerOomMemoryMarginPercent = "optimization.worker.oom.memory.margin-percent"
	// OptimizerWorkerOomMemoryMinIncrease is the jobmanagement key of oom worker memory margin percent
	OptimizerWorkerOomMemoryMinIncrease = "optimization.worker.oom.memory.min-increase"
	// OptimizerWorkerMemoryMarginPercent is the jobmanagement key of worker memory margin percent
	OptimizerWorkerMemoryMarginPercent = "optimization.worker.memory.margin-percent"
	// OptimizerWorkerMaxReplicaCount is the jobmanagement key of the maximum number of workers
	OptimizerWorkerMaxReplicaCount = "optimization.worker.max.replica-count"
	// OptimizerWorkerReplicaDecreaseCount is the jobmanagement key of worker replica decrease count each time
	OptimizerWorkerReplicaDecreaseCount = "optimization.worker.replica.decrease-count"
	// OptimizerWorkerMaxInitCountPerStep is the jobmanagement key of first max worker increase count
	OptimizerWorkerMaxInitCountPerStep = "optimization.worker.max.init-count-per-step"
	// OptimizerWorkerMaxCountPerStep is the jobmanagement key of max worker increase count per step
	OptimizerWorkerMaxCountPerStep = "optimization.worker.max.count-per-step"
	// OptimizerWorkerCPUMarginCore is the jobmanagement key of worker cpu margin cores
	OptimizerWorkerCPUMarginCore = "optimization.worker.cpu.margin-core"
	// OptimizerWorkerCPUUtilCompCount is the jobmanagement key of worker cpu util compare count
	OptimizerWorkerCPUUtilCompCount = "optimization.worker.cpu-util.comp-count"
	// OptimizerWorkerCPUUtilLessPercent is the jobmanagement key of worker cpu util less percent
	OptimizerWorkerCPUUtilLessPercent = "optimization.worker.cpu-util.less-percent"

	// OptimizerPSMemoryMarginPercent is the jobmanagement key of cold job initial ps memory
	OptimizerPSMemoryMarginPercent = "optimization.ps.memory.margin.percent"
	// OptimizerPSCPUMarginPercent is the jobmanagement key of cold job initial ps memory
	OptimizerPSCPUMarginPercent = "optimization.ps.cpu.margin.percent"
	// OptimizerPSMaxCount is the jobmanagement key of the max number of PS
	OptimizerPSMaxCount = "optimization.ps.count.maximum"
	// OptimizerPSMinCPUCore is the jobmanagement key of ps min cpu
	OptimizerPSMinCPUCore = "optimization.ps.min.cpu"
	// OptimizerPSColdMemory is the jobmanagement key of cold job initial ps memory
	OptimizerPSColdMemory = "optimization.ps.cold.memory"
	// OptimizerPSColdCPU is the jobmanagement key of cold job initial ps cpu
	OptimizerPSColdCPU = "optimization.ps.cold.cpu"
	// OptimizerPSColdReplica is the jobmanagement key of cold job initial ps replica
	OptimizerPSColdReplica = "optimization.ps.cold.replica"
	// OptimizerPSInitAdjustTargetWorkerCount is the target count of workers supporting by PS CPU.
	OptimizerPSInitAdjustTargetWorkerCount = "optimization.ps.init-adjust.target-worker-count"
	// OptimizerPSMarginCPU is the jobmanagement key of the ps margin cpu
	OptimizerPSMarginCPU = "optimization.ps.cpu.margin"
	// OptimizerPSCPUOverload is the jobmanagement key of ps cpu overloaded threshold
	OptimizerPSCPUOverload = "optimization.ps.cpu.overload"
	// OptimizerPSMemoryWorkloadUnbalancePercent is the jobmanagement key of ps memory workload unbalance percent
	OptimizerPSMemoryWorkloadUnbalancePercent = "optimization.ps.memory.workload-balance-percent"
	// OptimizerHotPSCPUThreshold is the jobmanagement key of ps overloaded threshold
	OptimizerHotPSCPUThreshold = "optimization.ps.cpu.hot-threshold"
	// OptimizerHotPSCPUTargetWorkerCount is the jobmanagement key of adjust overloaded ps cpu
	OptimizerHotPSCPUTargetWorkerCount = "optimization.ps.cpu.hot-target-worker-count"
	// OptimizerHotPSMemoryThreshold is the jobmanagement key of ps overloaded threshold
	OptimizerHotPSMemoryThreshold = "optimization.ps.memory.hot-threshold"
	// OptimizerHotPSMemoryAdjust is the jobmanagement key of adjust overloaded ps memory
	OptimizerHotPSMemoryAdjust = "optimization.ps.memory.hot-adjust"
	// OptimizerLowPSCPUThreshold is the jobmanagement key of ps overloaded threshold
	OptimizerLowPSCPUThreshold = "optimization.ps.cpu.low-threshold"
	// OptimizerPSCPUExhaustedThreshold is the jobmanagement key of ps cpu overloaded threshold
	OptimizerPSCPUExhaustedThreshold = "optimization.ps.cpu.exhausted-threshold"

	// OptimizerStepCountThreshold is the jobmanagement key of step count threshold
	OptimizerStepCountThreshold = "optimization.step.count.threshold"
	// OptimizerTrainingSpeedLessPercent is the jobmanagement key of training speed less comparison percent
	OptimizerTrainingSpeedLessPercent = "optimization.training-speed.less-percent"
	// OptimizerWorkerOptimizePhase is the jobmanagement key of worker optimize phase
	OptimizerWorkerOptimizePhase = "optimization.worker.optimize-phase"
	// OptimizerWorkerOptimizePhaseSample is the jobmanagement value of sample optimize phase
	OptimizerWorkerOptimizePhaseSample = "sample"
	// OptimizerWorkerOptimizePhaseInitial is the jobmanagement value of initial optimize phase
	OptimizerWorkerOptimizePhaseInitial = "initial"
	// OptimizerWorkerOptimizePhaseStable is the jobmanagement value of stable optimize phase
	OptimizerWorkerOptimizePhaseStable = "stable"

	// OptimizeRequestProcessorConfigMapName is the name of optimize request processor jobmanagement map
	OptimizeRequestProcessorConfigMapName = "optimize-request-processor.jobmanagement-map.name"
	// OptimizeRequestProcessorConfigMapKey is the key of optimize request processor jobmanagement map
	OptimizeRequestProcessorConfigMapKey = "optimize-request-processor.jobmanagement-map.key"
)

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
	// Namespace is the config key of namespace
	Namespace = "namespace"
	// KubeClientInterface is the config key of kube client
	KubeClientInterface = "kube.client.interface"
	// Cluster is the config key of cluster
	Cluster = "cluster"

	// DBUser is the config key of database user
	DBUser = "db.user"
	// DBPassword is the config key of database password
	DBPassword = "db.password"
	// DBEngineType is the config key of database engine type, e.g., mysql
	DBEngineType = "db.engine.type"
	// DBURL is the config key of database url
	DBURL = "db.url"

	// DataStoreConfigMapName is the name of data store config map
	DataStoreConfigMapName = "data-store.config-map.name"
	// DataStoreConfigMapKey is the key of data store config map
	DataStoreConfigMapKey = "data-store.config-map.key"
	// DataStoreName is the name of a data store
	DataStoreName = "data-store.name"

	// OptimizerConfigMapName is the name of optimizer config map
	OptimizerConfigMapName = "optimizer.config-map.name"
	// OptimizerConfigMapKey is the key of optimizer config map
	OptimizerConfigMapKey = "optimizer.config-map.key"

	// KubeWatcherConfigMapName is the name of kube watcher config map
	KubeWatcherConfigMapName = "kube-watcher.config-map.name"
	// KubeWatcherConfigMapKey is the key of kube watcher config map
	KubeWatcherConfigMapKey = "kube-watcher.config-map.key"
	// KubeWatcherMetricsAddress is the address of kube watcher metrics
	KubeWatcherMetricsAddress = "kube-watcher.metrics-address"
	// KubeWatcherEnableLeaderElect is key of if enable leader select of kube watcher
	KubeWatcherEnableLeaderElect = "kube-watcher.leader-elect.enable"
	// KubeWatcherHandlerConfigMapName is the name of kube watcher handler config map
	KubeWatcherHandlerConfigMapName = "kube-watcher.handler.config-map.name"
	// KubeWatcherHandlerConfigMapKey is the key of kube watcher handler config map
	KubeWatcherHandlerConfigMapKey = "kube-watcher.handler.config-map.key"

	// BrainServerConfigMapName is the name of brain server config map
	BrainServerConfigMapName = "brain.server.config-map.name"
	// BrainServerConfigMapKey is the name of brain server config key
	BrainServerConfigMapKey = "brain.server.config-map.key"

	// QueryBackwardTimePeriodInHour is the config key for query backward time period in hour
	QueryBackwardTimePeriodInHour = "query.backward.time-period.hour"

	// JobNodeMemoryMarginPercent is the config key for job node memory margin percent
	JobNodeMemoryMarginPercent = "job.node.memory.margin.percent"
	// JobNodeCPUMargin is the config key for job node cpu margin
	JobNodeCPUMargin = "job-node.cpu.margin"

	// OptimizerMinWorkerCreateCPU is the key of min CPU of the first worker
	OptimizerMinWorkerCreateCPU = "optimizer.worker.create.cpu.min"
	// OptimizerWorkerOomMemoryMarginPercent is the config key of oom worker memory margin percent
	OptimizerWorkerOomMemoryMarginPercent = "optimizer.worker.oom.memory.margin-percent"
	// OptimizerWorkerOomMemoryMinIncrease is the config key of oom worker memory margin percent
	OptimizerWorkerOomMemoryMinIncrease = "optimizer.worker.oom.memory.min-increase"
	// OptimizerWorkerMemoryMarginPercent is the config key of worker memory margin percent
	OptimizerWorkerMemoryMarginPercent = "optimizer.worker.memory.margin-percent"
	// OptimizerWorkerMaxReplicaCount is the config key of the maximum number of workers
	OptimizerWorkerMaxReplicaCount = "optimizer.worker.max.replica-count"
	// OptimizerWorkerReplicaDecreaseCount is the config key of worker replica decrease count each time
	OptimizerWorkerReplicaDecreaseCount = "optimizer.worker.replica.decrease-count"
	// OptimizerWorkerMaxInitCountPerStep is the config key of first max worker increase count
	OptimizerWorkerMaxInitCountPerStep = "optimizer.worker.max.init-count-per-step"
	// OptimizerWorkerMaxCountPerStep is the config key of max worker increase count per step
	OptimizerWorkerMaxCountPerStep = "optimizer.worker.max.count-per-step"
	// OptimizerWorkerCPUMarginCore is the config key of worker cpu margin cores
	OptimizerWorkerCPUMarginCore = "optimizer.worker.cpu.margin-core"
	// OptimizerWorkerCPUUtilCompCount is the config key of worker cpu util compare count
	OptimizerWorkerCPUUtilCompCount = "optimizer.worker.cpu-util.comp-count"
	// OptimizerWorkerCPUUtilLessPercent is the config key of worker cpu util less percent
	OptimizerWorkerCPUUtilLessPercent = "optimizer.worker.cpu-util.less-percent"

	// OptimizerPSMemoryMarginPercent is the config key of cold job initial ps memory
	OptimizerPSMemoryMarginPercent = "optimizer.ps.memory.margin.percent"
	// OptimizerPSCPUMarginPercent is the config key of cold job initial ps memory
	OptimizerPSCPUMarginPercent = "optimizer.ps.cpu.margin.percent"
	// OptimizerPSMaxCount is the config key of the max number of PS
	OptimizerPSMaxCount = "optimizer.ps.count.maximum"
	// OptimizerPSMinCPUCore is the config key of ps min cpu
	OptimizerPSMinCPUCore = "optimizer.ps.min.cpu"
	// OptimizerPSColdMemory is the config key of cold job initial ps memory
	OptimizerPSColdMemory = "optimizer.ps.cold.memory"
	// OptimizerPSColdCPU is the config key of cold job initial ps cpu
	OptimizerPSColdCPU = "optimizer.ps.cold.cpu"
	// OptimizerPSColdReplica is the config key of cold job initial ps replica
	OptimizerPSColdReplica = "optimizer.ps.cold.replica"
	// OptimizerPSInitAdjustTargetWorkerCount is the target count of workers supporting by PS CPU.
	OptimizerPSInitAdjustTargetWorkerCount = "optimizer.ps.init-adjust.target-worker-count"
	// OptimizerPSMarginCPU is the config key of the ps margin cpu
	OptimizerPSMarginCPU = "optimizer.ps.cpu.margin"
	// OptimizerPSCPUOverload is the config key of ps cpu overloaded threshold
	OptimizerPSCPUOverload = "optimizer.ps.cpu.overload"
	// OptimizerPSMemoryWorkloadUnbalancePercent is the config key of ps memory workload unbalance percent
	OptimizerPSMemoryWorkloadUnbalancePercent = "optimizer.ps.memory.workload-balance-percent"
	// OptimizerHotPSCPUThreshold is the config key of ps overloaded threshold
	OptimizerHotPSCPUThreshold = "optimizer.ps.cpu.hot-threshold"
	// OptimizerHotPSCPUTargetWorkerCount is the config key of adjust overloaded ps cpu
	OptimizerHotPSCPUTargetWorkerCount = "optimizer.ps.cpu.hot-target-worker-count"
	// OptimizerHotPSMemoryThreshold is the config key of ps overloaded threshold
	OptimizerHotPSMemoryThreshold = "optimizer.ps.memory.hot-threshold"
	// OptimizerHotPSMemoryAdjust is the config key of adjust overloaded ps memory
	OptimizerHotPSMemoryAdjust = "optimizer.ps.memory.hot-adjust"
	// OptimizerLowPSCPUThreshold is the config key of ps overloaded threshold
	OptimizerLowPSCPUThreshold = "optimizer.ps.cpu.low-threshold"
	// OptimizerPSCPUExhaustedThreshold is the config key of ps cpu overloaded threshold
	OptimizerPSCPUExhaustedThreshold = "optimizer.ps.cpu.exhausted-threshold"

	// OptimizerStepCountThreshold is the config key of step count threshold
	OptimizerStepCountThreshold = "optimizer.step.count.threshold"
	// OptimizerTrainingSpeedLessPercent is the config key of training speed less comparison percent
	OptimizerTrainingSpeedLessPercent = "optimizer.training-speed.less-percent"
	// OptimizerWorkerOptimizePhase is the config key of worker optimize phase
	OptimizerWorkerOptimizePhase = "optimizer.worker.optimize-phase"
	// OptimizerWorkerOptimizePhaseSample is the config value of sample optimize phase
	OptimizerWorkerOptimizePhaseSample = "sample"
	// OptimizerWorkerOptimizePhaseInitial is the config value of initial optimize phase
	OptimizerWorkerOptimizePhaseInitial = "initial"
	// OptimizerWorkerOptimizePhaseStable is the config value of stable optimize phase
	OptimizerWorkerOptimizePhaseStable = "stable"

	// OptimizeRequestProcessorConfigMapName is the name of optimize request processor config map
	OptimizeRequestProcessorConfigMapName = "optimize-request-processor.config-map.name"
	// OptimizeRequestProcessorConfigMapKey is the key of optimize request processor config map
	OptimizeRequestProcessorConfigMapKey = "optimize-request-processor.config-map.key"
)

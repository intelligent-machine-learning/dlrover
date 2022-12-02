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
	// KubeWatcherRegisterCreateEventAdmin is the config key of administrator which register create events
	KubeWatcherRegisterCreateEventAdmin = "kube-watcher.register-create-admin"
)

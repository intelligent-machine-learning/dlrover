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

package watcher

import (
	"context"
	"fmt"
	log "github.com/golang/glog"
	"github.com/intelligent-machine-learning/easydl/brain/pkg/common"
	"github.com/intelligent-machine-learning/easydl/brain/pkg/config"
	"github.com/intelligent-machine-learning/easydl/brain/pkg/datastore"
	handlerimpl "github.com/intelligent-machine-learning/easydl/brain/pkg/platform/k8s/implementation/watchhandler"
	watchercommon "github.com/intelligent-machine-learning/easydl/brain/pkg/platform/k8s/watcher/common"
	elasticv1alpha1 "github.com/intelligent-machine-learning/easydl/dlrover/go/operator/api/v1alpha1"
	"k8s.io/client-go/kubernetes"
	k8smanager "sigs.k8s.io/controller-runtime/pkg/manager"
	"sync"
)

const (
	logName = "k8s-watcher"
)

// Manager is the struct of watcher manager
type Manager struct {
	watcherConf          *config.Config
	watcherConfigManager *config.Manager

	handlerConf          *config.Config
	handlerConfigManager *config.Manager

	dsManager *datastore.Manager

	kubeClientSet kubernetes.Interface

	watchHandlerRegisterFuncs map[string]handlerimpl.RegisterWatchHandlerFunc
	managerMutex              *sync.Mutex
	kubeWatcher               *watchercommon.KubeWatcher
	job                       *elasticv1alpha1.ElasticJob
}

// NewManager returns a new kube watcher manager
func NewManager(conf *config.Config) (*Manager, error) {
	namespace := conf.GetString(config.Namespace)
	configMapName := conf.GetString(config.KubeWatcherConfigMapName)
	configMapKey := conf.GetString(config.KubeWatcherConfigMapKey)
	kubeClientSet := conf.GetKubeClientInterface()

	manager := &Manager{
		watcherConfigManager:      config.NewManager(namespace, configMapName, configMapKey, kubeClientSet),
		kubeClientSet:             kubeClientSet,
		watchHandlerRegisterFuncs: handlerimpl.GetRegisterWatchHandlerFuncs(),
		managerMutex:              &sync.Mutex{},
	}
	return manager, nil
}

// CleanUp cleans up the manager
func (manager *Manager) CleanUp() error {
	return nil
}

// Run starts the manager
func (manager *Manager) Run(ctx context.Context, errReporter common.ErrorReporter, errHandler common.ErrorHandler) error {
	err := manager.watcherConfigManager.Run(ctx, errReporter)
	if err != nil {
		err = fmt.Errorf("[%s] failed to initialize watch handler config manager: %v", logName, err)
		log.Error(err)
		return err
	}
	manager.watcherConf, err = manager.watcherConfigManager.GetConfig()
	if err != nil {
		log.Errorf("[%s] fail to get watch handler config: %v", logName, err)
		return err
	}
	log.Infof("[%s] k8s watcher config: %v", logName, manager.watcherConf)

	opts := watchercommon.KubeWatchOptions{
		Options: k8smanager.Options{
			MetricsBindAddress:      manager.watcherConf.GetString(config.KubeWatcherMetricsAddress),
			LeaderElection:          manager.watcherConf.GetBool(config.KubeWatcherEnableLeaderElect),
			LeaderElectionID:        fmt.Sprintf("%s-k8s-watcher-mananger-leader", manager.watcherConf.GetString(config.Namespace)),
			LeaderElectionNamespace: manager.watcherConf.GetString(config.Namespace),
			Namespace:               manager.watcherConf.GetString(config.Namespace),
		},
	}

	manager.kubeWatcher, err = watchercommon.NewKubeWatcher(ctx, errHandler, opts)
	if err != nil {
		log.Errorf("[%s] failed to new the kubeWatch with err: %v", logName, err)
		return err
	}

	// Start the kubeWatcher
	if err = manager.kubeWatcher.Start(); err != nil {
		log.Errorf("[%s] failed to start the kubeWatch with err: %v", logName, err)
		return err
	}

	dsConf := config.NewEmptyConfig()
	dsConf.Set(config.KubeClientInterface, manager.kubeClientSet)
	dsConf.Set(config.DataStoreConfigMapName, manager.watcherConf.GetString(config.DataStoreConfigMapName))
	dsConf.Set(config.DataStoreConfigMapKey, manager.watcherConf.GetString(config.DataStoreConfigMapKey))
	dsConf.Set(config.Namespace, manager.watcherConf.GetString(config.Namespace))

	manager.dsManager = datastore.NewManager(dsConf)
	err = manager.dsManager.Run(ctx, errReporter)
	if err != nil {
		log.Errorf("[%s] fail to run the data store manager: %v", logName, err)
		return err
	}

	namespace := manager.watcherConf.GetString(config.Namespace)
	configMapName := manager.watcherConf.GetString(config.KubeWatcherHandlerConfigMapName)
	configMapKey := manager.watcherConf.GetString(config.KubeWatcherHandlerConfigMapKey)
	kubeClientSet := manager.kubeClientSet
	manager.handlerConfigManager = config.NewManager(namespace, configMapName, configMapKey, kubeClientSet)
	err = manager.handlerConfigManager.Run(ctx, errReporter)
	if err != nil {
		log.Errorf("[%s] fail to run k8s watcher handler config manager: %v", logName, err)
		return err
	}

	manager.handlerConf, err = manager.handlerConfigManager.GetConfig()
	if err != nil {
		log.Errorf("[%s] fail to get k8s watcher handler config: %v", logName, err)
		return err
	}

	manager.registerWatchHandlers(ctx)
	return nil
}

// Create and start a watch handler
func (manager *Manager) createAndRegisterWatchHandler(name string, conf *config.Config) error {
	dataStoreName := conf.GetString(config.DataStoreName)
	dataStore, err := manager.dsManager.CreateDataStore(dataStoreName)
	if err != nil {
		log.Errorf("[%s] fail to create data store %s for watcher handler %s: %v", logName, dataStoreName, name, err)
		return err
	}
	registerFunc, found := manager.watchHandlerRegisterFuncs[name]
	if !found {
		err = fmt.Errorf("watcher handler %s does not have a register func", name)
		log.Error(err)
		return err
	}
	err = registerFunc(manager.kubeWatcher, name, conf, dataStore)
	if err != nil {
		return err
	}

	return nil
}

// Update watch handler according to the configuration
func (manager *Manager) registerWatchHandlers(ctx context.Context) error {
	manager.managerMutex.Lock()
	defer manager.managerMutex.Unlock()

	manager.registerSchema()
	watchHandlerNames := manager.handlerConf.GetKeys()
	for _, watchHandlerName := range watchHandlerNames {
		watchHandlerConfig := manager.handlerConf.GetConfig(watchHandlerName)
		if watchHandlerConfig == nil {
			log.Errorf("Fail to get the config for watch handler %s", watchHandlerName)
			continue
		}
		log.Infof("watch handler %s config: %v", watchHandlerName, watchHandlerConfig)

		if err := manager.createAndRegisterWatchHandler(watchHandlerName, watchHandlerConfig); err != nil {
			log.Errorf("Fail to create and register WatchHandler %s: %v", watchHandlerName, err)
			continue
		}
	}

	return nil
}

func (manager *Manager) registerSchema() {
	manager.kubeWatcher.RegisterKubeResourceScheme(elasticv1alpha1.SchemeBuilder)
}

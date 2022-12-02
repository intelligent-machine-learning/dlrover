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
	k8smanager "sigs.k8s.io/controller-runtime/pkg/manager"
	"sync"
)

// Manager is the struct of watcher manager
type Manager struct {
	conf                      *config.Config
	handlerConf               *config.Config
	handlerConfigManager      *config.Manager
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

	configManager := config.NewManager(namespace, configMapName, configMapKey, kubeClientSet)

	manager := &Manager{
		conf:                      conf,
		watchHandlerRegisterFuncs: handlerimpl.GetRegisterWatchHandlerFuncs(),
		handlerConfigManager:      configManager,
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
	err := manager.handlerConfigManager.Run(ctx, errReporter)
	if err != nil {
		err = fmt.Errorf("failed to initialize watch handler config manager: %v", err)
		log.Error(err)
		return err
	}
	manager.handlerConf, err = manager.handlerConfigManager.GetConfig()
	if err != nil {
		log.Errorf("fail to get watch handler config: %v", err)
		return err
	}

	opts := watchercommon.KubeWatchOptions{
		Options: k8smanager.Options{
			MetricsBindAddress:      manager.conf.GetString(config.KubeWatcherMetricsAddress),
			LeaderElection:          manager.conf.GetBool(config.KubeWatcherEnableLeaderElect),
			LeaderElectionID:        fmt.Sprintf("%s-k8sResWatchHandlerManager-leader", manager.conf.GetString(config.Namespace)),
			LeaderElectionNamespace: manager.conf.GetString(config.Namespace),
			Namespace:               manager.conf.GetString(config.Namespace),
		},
	}

	manager.kubeWatcher, err = watchercommon.NewKubeWatcher(ctx, errHandler, opts)
	if err != nil {
		log.Errorf("Failed to new the kubeWatch with err: %v", err)
		return err
	}

	// Start the kubeWatcher
	if err = manager.kubeWatcher.Start(); err != nil {
		log.Errorf("Failed to start the kubeWatch with err: %v", err)
		return err
	}

	manager.registerWatchHandlers(ctx, errReporter)
	return nil
}

// Create and start a watch handler
func (manager *Manager) createAndRegisterWatchHandler(ctx context.Context, dataStoreManager *datastore.Manager, name string, conf *config.Config) error {
	dataStoreName := conf.GetString(config.DataStoreName)
	dataStore, err := dataStoreManager.CreateDataStore(dataStoreName)
	if err != nil {
		log.Errorf("fail to create data store %s for watcher handler %s: %v", dataStoreName, name, err)
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
func (manager *Manager) registerWatchHandlers(ctx context.Context, errReporter common.ErrorReporter) error {
	log.Infof("Update watch handlers based on config: %v", manager.conf)
	manager.managerMutex.Lock()
	defer manager.managerMutex.Unlock()

	dsManager := datastore.NewManager(manager.conf)
	err := dsManager.Run(ctx, errReporter)
	if err != nil {
		log.Errorf("fail to run the data store mananger: %v", err)
		return err
	}

	manager.registerSchema()
	watchHandlerNames := manager.handlerConf.GetKeys()
	for _, watchHandlerName := range watchHandlerNames {
		watchHandlerConfig := manager.handlerConf.GetConfig(watchHandlerName)
		if watchHandlerConfig == nil {
			log.Errorf("Fail to get the config for watch handler %s", watchHandlerName)
			continue
		}

		if err = manager.createAndRegisterWatchHandler(ctx, dsManager, watchHandlerName, watchHandlerConfig); err != nil {
			log.Errorf("Fail to create and register WatchHandler %s: %v", watchHandlerName, err)
			continue
		}
	}

	return nil
}

func (manager *Manager) registerSchema() {
	manager.kubeWatcher.RegisterKubeResourceScheme(elasticv1alpha1.SchemeBuilder)
}

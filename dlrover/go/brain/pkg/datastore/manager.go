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

package datastore

import (
	"context"
	"fmt"
	log "github.com/golang/glog"
	"github.com/intelligent-machine-learning/easydl/brain/pkg/common"
	"github.com/intelligent-machine-learning/easydl/brain/pkg/config"
	datastoreapi "github.com/intelligent-machine-learning/easydl/brain/pkg/datastore/api"
	dsimpl "github.com/intelligent-machine-learning/easydl/brain/pkg/datastore/implementation"
	"sync"
)

// Manager is the struct of data store manager
type Manager struct {
	conf          *config.Config
	configManager *config.Manager
	dataStores    map[string]datastoreapi.DataStore
	mutex         *sync.RWMutex
}

// NewManager creates a data store manager
func NewManager(conf *config.Config) *Manager {
	namespace := conf.GetString(config.Namespace)
	configMapName := conf.GetString(config.DataStoreConfigMapName)
	configMapKey := conf.GetString(config.DataStoreConfigMapKey)
	kubeClientSet := conf.GetKubeClientInterface()

	configManager := config.NewManager(namespace, configMapName, configMapKey, kubeClientSet)

	return &Manager{
		configManager: configManager,
		dataStores:    make(map[string]datastoreapi.DataStore),
		mutex:         &sync.RWMutex{},
	}
}

// Run starts data store manager
func (m *Manager) Run(ctx context.Context, errReporter common.ErrorReporter) error {
	if err := m.configManager.Run(ctx, errReporter); err != nil {
		log.Errorf("[DataStore Manager] fail to run config manager: %v", err)
		return err
	}

	conf, err := m.configManager.GetConfig()
	if err != nil {
		log.Errorf("[DataStore Manager] fail to get data store config: %v", err)
		return err
	}

	m.conf = conf
	return nil
}

// CreateDataStore creates a data store for a given name
func (m *Manager) CreateDataStore(name string) (datastoreapi.DataStore, error) {
	m.mutex.Lock()
	defer m.mutex.Unlock()

	if dataStore, found := m.dataStores[name]; found {
		return dataStore, nil
	}

	conf := m.conf.GetConfig(name)
	if conf == nil {
		err := fmt.Errorf("There is no config for data store %s", name)
		return nil, err
	}

	dataStore, err := dsimpl.CreateDataStore(name, conf)
	if err != nil {
		log.Errorf("fail to create data store %s: %v", name, err)
		return nil, err
	}
	m.dataStores[name] = dataStore
	return dataStore, nil
}

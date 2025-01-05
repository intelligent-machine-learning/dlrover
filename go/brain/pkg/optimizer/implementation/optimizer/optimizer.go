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

package optimizer

import (
	"context"
	"fmt"
	log "github.com/golang/glog"
	"github.com/intelligent-machine-learning/easydl/brain/pkg/common"
	"github.com/intelligent-machine-learning/easydl/brain/pkg/config"
	"github.com/intelligent-machine-learning/easydl/brain/pkg/datastore"
	datastoreapi "github.com/intelligent-machine-learning/easydl/brain/pkg/datastore/api"
	"github.com/intelligent-machine-learning/easydl/brain/pkg/optimizer/api"
	optconfig "github.com/intelligent-machine-learning/easydl/brain/pkg/optimizer/config"
	"sync"
)

const (
	defaultStepCountThreshold = 5
	defaultPSCPUOverload      = 0.8

	loggerName = "Optimizer Manager"
)

var (
	locker            = &sync.RWMutex{}
	newOptimizerFuncs = make(map[string]newOptimizerFunc)
)

type newOptimizerFunc func(dataStore datastoreapi.DataStore, config *config.Config) api.Optimizer

func registerNewOptimizerFunc(name string, newFunc newOptimizerFunc) error {
	locker.Lock()
	defer locker.Unlock()

	if _, found := newOptimizerFuncs[name]; found {
		err := fmt.Errorf("NewOptimizerFunc %s has already registered", name)
		return err
	}
	newOptimizerFuncs[name] = newFunc
	return nil
}

func createOptimizer(name string, dataStore datastoreapi.DataStore, conf *config.Config) (api.Optimizer, error) {
	locker.Lock()
	defer locker.Unlock()

	newFunc, exist := newOptimizerFuncs[name]
	if !exist {
		err := fmt.Errorf("Optimizer %s has not registered", name)
		return nil, err
	}
	optimizer := newFunc(dataStore, conf)
	return optimizer, nil
}

// Manager is the struct of optimizer manager
type Manager struct {
	optimizers    map[string]api.Optimizer
	dsManager     *datastore.Manager
	conf          *config.Config
	configManager *config.Manager
	locker        *sync.RWMutex
}

// NewManager creates a new optimizer manager
func NewManager(conf *config.Config, dsManager *datastore.Manager) *Manager {
	namespace := conf.GetString(config.Namespace)
	configMapName := conf.GetString(config.OptimizerConfigMapName)
	configMapKey := conf.GetString(config.OptimizerConfigMapKey)
	kubeClientSet := conf.GetKubeClientInterface()

	return &Manager{
		configManager: config.NewManager(namespace, configMapName, configMapKey, kubeClientSet),
		dsManager:     dsManager,
		locker:        &sync.RWMutex{},
	}
}

// Run starts the manager
func (m *Manager) Run(ctx context.Context, errReporter common.ErrorReporter) error {
	var err error

	if err = m.configManager.Run(ctx, errReporter); err != nil {
		log.Errorf("[Optimizer Manager] fail to run optimizer config manager: %v", err)
		return err
	}

	if m.conf, err = m.configManager.GetConfig(); err != nil {
		log.Errorf("[%s] fail to get optimizer config: %v", loggerName, err)
		return err
	}
	m.configManager.RegisterConfigObserver("dlrover-optimizers", m.OptimizersConfigUpdateNotify)

	m.locker.Lock()
	defer m.locker.Unlock()

	m.optimizers = createOptimizers(m.conf, m.dsManager)
	log.Infof("[%s] create optimizers: %v", loggerName, m.optimizers)
	return nil
}

// Optimize process OptimizeRequest
func (m *Manager) Optimize(conf *optconfig.OptimizerConfig, jobs []*common.JobMeta) ([]*common.OptimizePlan, error) {
	m.locker.RLock()
	defer m.locker.RUnlock()

	optimizer, _ := m.optimizers[conf.Name]
	if optimizer == nil {
		err := fmt.Errorf("Optimizer %s does not register", conf.Name)
		return nil, err
	}

	plans, err := optimizer.Optimize(conf, jobs)
	if err != nil {
		return nil, err
	}
	return plans, nil
}

// OptimizersConfigUpdateNotify update optimizers when observe the config is updated
func (m *Manager) OptimizersConfigUpdateNotify(newConf *config.Config) error {
	m.locker.Lock()
	defer m.locker.Unlock()

	log.Infof("[%s] optimizers new conf: %v", loggerName, newConf)
	m.conf = newConf
	m.optimizers = createOptimizers(newConf, m.dsManager)
	return nil
}

func createOptimizers(optimizersConf *config.Config, dsManager *datastore.Manager) map[string]api.Optimizer {
	log.Infof("create optimizers for config: %v", optimizersConf)
	optimizerNames := optimizersConf.GetKeys()
	optimizers := make(map[string]api.Optimizer)
	for _, name := range optimizerNames {
		conf := optimizersConf.GetConfig(name)
		if conf == nil {
			log.Errorf("[%s] no config for optimizer %s", loggerName, name)
			continue
		}
		dataStoreName := conf.GetString(config.DataStoreName)
		dataStore, err := dsManager.CreateDataStore(dataStoreName)
		if err != nil {
			log.Warningf("[Optimizer Manager] fail to get data store %s for %s: %v", dataStoreName, name, err)
			dataStore = nil
		}
		optimizer, err := createOptimizer(name, dataStore, conf)
		if err != nil {
			log.Errorf("[%s] fail to create optimizer %s: %v", loggerName, name, err)
			continue
		}
		optimizers[name] = optimizer
	}
	return optimizers
}

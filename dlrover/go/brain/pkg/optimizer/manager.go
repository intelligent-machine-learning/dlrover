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
	optapi "github.com/intelligent-machine-learning/easydl/brain/pkg/optimizer/api"
	optimpl "github.com/intelligent-machine-learning/easydl/brain/pkg/optimizer/implementation/optimizer"
	pb "github.com/intelligent-machine-learning/easydl/brain/pkg/proto"
	"github.com/intelligent-machine-learning/easydl/brain/pkg/utils"
	"sync"
)

// Manager is the struct to manage optimizers
type Manager struct {
	optimizers           map[string]optapi.Optimizer
	dataStoreManager     *datastore.Manager
	confRetrieverManager *ConfigRetrieverManager
	optimizerConfig      *config.Config
	configManager        *config.Manager
	locker               *sync.RWMutex
}

// NewManager creates a new OptimizerManager
func NewManager(conf *config.Config) *Manager {
	namespace := conf.GetString(config.Namespace)
	configMapName := conf.GetString(config.OptimizerConfigMapName)
	configMapKey := conf.GetString(config.OptimizerConfigMapKey)
	kubeClientSet := conf.GetKubeClientInterface()

	configManager := config.NewManager(namespace, configMapName, configMapKey, kubeClientSet)

	return &Manager{
		configManager:        configManager,
		dataStoreManager:     datastore.NewManager(conf),
		confRetrieverManager: NewConfigRetrieverManager(conf),
		locker:               &sync.RWMutex{},
	}
}

// Run starts the manager
func (m *Manager) Run(ctx context.Context, errReporter common.ErrorReporter) error {
	if err := m.configManager.Run(ctx, errReporter); err != nil {
		log.Errorf("[Optimizer Manager] fail to run optimizer config manager: %v", err)
		return err
	}

	optimizerConfig, err := m.configManager.GetConfig()
	if err != nil {
		log.Errorf("[Optimizer Manager] fail to get optimizer config: %v", err)
		return err
	}
	m.optimizerConfig = optimizerConfig

	m.configManager.RegisterConfigObserver("easydl-optimizers", m.OptimizersConfigUpdateNotify)

	err = m.dataStoreManager.Run(ctx, errReporter)
	if err != nil {
		log.Errorf("fail to run the data store manager: %v", err)
		return err
	}

	m.locker.Lock()
	defer m.locker.Unlock()

	m.optimizers = createOptimizers(m.optimizerConfig, m.dataStoreManager)
	log.Infof("create optimizers: %v", m.optimizers)
	return nil
}

// Optimize process OptimizeRequest
func (m *Manager) Optimize(request *pb.OptimizeRequest) ([]*pb.JobOptimizePlan, error) {
	m.locker.RLock()
	defer m.locker.RUnlock()

	if request == nil {
		err := fmt.Errorf("Error optimize config: %v", request)
		return nil, err
	}

	optimizerConfig, err := m.confRetrieverManager.RetrieveOptimizerConfig(request.Config)
	if err != nil {
		log.Errorf("Fail to retrieve optimizer config from %v: %v", request, err)
		return nil, err
	}

	optimizer, _ := m.optimizers[optimizerConfig.Name]
	if optimizer == nil {
		err = fmt.Errorf("Optimizer %s does not register", optimizerConfig.Name)
		return nil, err
	}

	jobs := make([]*common.JobMeta, 0)
	for _, pbJob := range request.Jobs {
		job := utils.ConvertPBOptimizeJobMetaToJobMeta(pbJob)
		jobs = append(jobs, job)
	}

	optimizePlans, err := optimizer.Optimize(optimizerConfig, jobs)
	if err != nil {
		return nil, err
	}

	plans := make([]*pb.JobOptimizePlan, 0)
	for _, optimizePlan := range optimizePlans {
		plan := utils.ConvertOptimizePlanToPBJobOptimizePlan(optimizePlan)
		plans = append(plans, plan)
	}
	return plans, nil
}

// OptimizersConfigUpdateNotify update optimizers when observe the config is updated
func (m *Manager) OptimizersConfigUpdateNotify(newConf *config.Config) error {
	m.locker.Lock()
	defer m.locker.Unlock()

	log.Infof("optimizers new conf: %v", newConf)
	m.optimizerConfig = newConf
	m.optimizers = createOptimizers(newConf, m.dataStoreManager)
	return nil
}

func createOptimizers(optimizersConf *config.Config, dsManager *datastore.Manager) map[string]optapi.Optimizer {
	log.Infof("create optimizers for config: %v", optimizersConf)
	optimizerNames := optimizersConf.GetKeys()
	optimizers := make(map[string]optapi.Optimizer)
	for _, name := range optimizerNames {
		conf := optimizersConf.GetConfig(name)
		if conf == nil {
			log.Errorf("[Optimizer Manager] no config for optimizer %s", name)
			continue
		}
		dataStoreName := conf.GetString(config.DataStoreName)
		dataStore, err := dsManager.CreateDataStore(dataStoreName)
		if err != nil {
			log.Warningf("[Optimizer Manager] fail to get data store %s for %s: %v", dataStoreName, name, err)
			dataStore = nil
		}
		optimizer, err := optimpl.CreateOptimizer(name, dataStore, conf)
		if err != nil {
			log.Errorf("[Optimizer Manager] fail to create optimizer %s: %v", name, err)
			continue
		}
		optimizers[name] = optimizer
	}
	return optimizers
}

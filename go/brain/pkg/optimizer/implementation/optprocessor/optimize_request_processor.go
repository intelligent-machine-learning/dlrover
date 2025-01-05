// Copyright 2023 The DLRover Authors. All rights reserved.
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

package optprocessor

import (
	"context"
	"fmt"
	log "github.com/golang/glog"
	"github.com/intelligent-machine-learning/easydl/brain/pkg/common"
	"github.com/intelligent-machine-learning/easydl/brain/pkg/config"
	"github.com/intelligent-machine-learning/easydl/brain/pkg/datastore"
	"github.com/intelligent-machine-learning/easydl/brain/pkg/optimizer/api"
	optimizercommon "github.com/intelligent-machine-learning/easydl/brain/pkg/optimizer/common"
	implconfigretriever "github.com/intelligent-machine-learning/easydl/brain/pkg/optimizer/implementation/configretriever"
	imploptimizer "github.com/intelligent-machine-learning/easydl/brain/pkg/optimizer/implementation/optimizer"
	pb "github.com/intelligent-machine-learning/easydl/brain/pkg/proto"
	"github.com/intelligent-machine-learning/easydl/brain/pkg/utils"
	"sync"
)

const (
	loggerName = "OptimizeRequestProcessor Manager"
)

var (
	locker                           = &sync.RWMutex{}
	newOptimizeRequestProcessorFuncs = make(map[string]newOptimizeRequestProcessorFunc)
)

type newOptimizeRequestProcessorFunc func(conf *config.Config, dsManager *datastore.Manager,
	optimizerManager *imploptimizer.Manager) api.OptimizeRequestProcessor

func registerNewOptimizeRequestOptimizerFunc(name string, newFunc newOptimizeRequestProcessorFunc) error {
	locker.Lock()
	defer locker.Unlock()

	if _, found := newOptimizeRequestProcessorFuncs[name]; found {
		err := fmt.Errorf("NewOptimizeRequestOptimizeFunc %s has already registered", name)
		return err
	}
	newOptimizeRequestProcessorFuncs[name] = newFunc
	return nil
}

// Manager is the struct of optimize request processor
type Manager struct {
	processors             map[string]api.OptimizeRequestProcessor
	processorConfig        *config.Config
	processorConfigManager *config.Manager

	dsManager              *datastore.Manager
	optimizerManager       *imploptimizer.Manager
	configRetrieverManager *implconfigretriever.Manager

	locker *sync.RWMutex
}

// NewManager creates a new optimize request processor manager
func NewManager(conf *config.Config) *Manager {
	namespace := conf.GetString(config.Namespace)
	configMapName := conf.GetString(config.OptimizeRequestProcessorConfigMapName)
	configMapKey := conf.GetString(config.OptimizeRequestProcessorConfigMapKey)
	kubeClientSet := conf.GetKubeClientInterface()

	dsManager := datastore.NewManager(conf)

	return &Manager{
		processors:             make(map[string]api.OptimizeRequestProcessor),
		processorConfigManager: config.NewManager(namespace, configMapName, configMapKey, kubeClientSet),
		dsManager:              dsManager,
		configRetrieverManager: implconfigretriever.NewConfigRetrieverManager(nil),
		optimizerManager:       imploptimizer.NewManager(conf, dsManager),
		locker:                 &sync.RWMutex{},
	}
}

// Run starts the manager
func (m *Manager) Run(ctx context.Context, errReporter common.ErrorReporter) error {
	var err error

	if err = m.processorConfigManager.Run(ctx, errReporter); err != nil {
		log.Errorf("[%s] fail to run optimize request processor config manager: %v", loggerName, err)
		return err
	}
	if m.processorConfig, err = m.processorConfigManager.GetConfig(); err != nil {
		log.Errorf("[%s] fail to get processor config: %v", loggerName, err)
		return err
	}
	m.processorConfigManager.RegisterConfigObserver("dlrover-optimize-request-processor", m.OptimizeRequestProcessorConfigUpdateNotify)

	if err = m.dsManager.Run(ctx, errReporter); err != nil {
		log.Errorf("[%s] fail to run data store manager: %v", loggerName, err)
		return err
	}

	if err = m.optimizerManager.Run(ctx, errReporter); err != nil {
		log.Errorf("[%s] fail to run optimizer manager: %v", loggerName, err)
		return err
	}

	m.locker.Lock()
	defer m.locker.Unlock()

	m.processors = createOptimizeRequestProcessors(m.processorConfig, m.dsManager, m.optimizerManager)
	return nil
}

// ProcessOptimizeRequest processes optimize request
func (m *Manager) ProcessOptimizeRequest(ctx context.Context, request *pb.OptimizeRequest) ([]*pb.JobOptimizePlan, error) {
	m.locker.RLock()
	defer m.locker.RUnlock()

	if request == nil {
		err := fmt.Errorf("empty optimize request: %v", request)
		return nil, err
	}

	optimizerConfig, err := m.configRetrieverManager.RetrieveOptimizerConfig(request.Config)
	if err != nil {
		log.Errorf("[%s] fail to retrieve optimizer config from %v: %v", loggerName, request, err)
		return nil, err
	}

	jobs := make([]*common.JobMeta, 0)
	for _, pbJob := range request.Jobs {
		job := utils.ConvertPBOptimizeJobMetaToJobMeta(pbJob)
		jobs = append(jobs, job)
	}

	processorName := request.Config.BrainProcessor
	event := &optimizercommon.OptimizeEvent{
		Type:          request.Type,
		ProcessorName: processorName,
		DataStoreName: request.Config.DataStore,
		Jobs:          jobs,
		Conf:          optimizerConfig,
	}

	processor, _ := m.processors[processorName]
	if processor == nil {
		err = fmt.Errorf("Processor %s does not register", processorName)
		return nil, err
	}

	optimizePlans, err := processor.Optimize(ctx, event)
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

// OptimizeRequestProcessorConfigUpdateNotify update processors when observe the config is updated
func (m *Manager) OptimizeRequestProcessorConfigUpdateNotify(newConf *config.Config) error {
	m.locker.Lock()
	defer m.locker.Unlock()

	log.Infof("[%s] optimize request processor new conf: %v", loggerName, newConf)
	m.processorConfig = newConf
	m.processors = createOptimizeRequestProcessors(newConf, m.dsManager, m.optimizerManager)
	return nil
}

func createOptimizeRequestProcessor(name string, conf *config.Config, dsManager *datastore.Manager,
	optimizerManager *imploptimizer.Manager) (api.OptimizeRequestProcessor, error) {

	locker.Lock()
	defer locker.Unlock()

	newFunc, exist := newOptimizeRequestProcessorFuncs[name]
	if !exist {
		err := fmt.Errorf("Optimize request processor %s has not registered", name)
		return nil, err
	}
	processor := newFunc(conf, dsManager, optimizerManager)
	return processor, nil
}

func createOptimizeRequestProcessors(conf *config.Config, dsManager *datastore.Manager,
	optimizerManager *imploptimizer.Manager) map[string]api.OptimizeRequestProcessor {
	log.Infof("[%s] create optimize request processors according to config: %v", loggerName, conf)
	processorNames := conf.GetKeys()
	processors := make(map[string]api.OptimizeRequestProcessor)
	for _, name := range processorNames {
		processorConfig := conf.GetConfig(name)
		if conf == nil {
			log.Errorf("[%s] no config for optimize request processor %s", loggerName, name)
			continue
		}
		processor, err := createOptimizeRequestProcessor(name, processorConfig, dsManager, optimizerManager)
		if err != nil {
			log.Errorf("[%s] fail to create optimize request processor %s: %v", loggerName, name, err)
			continue
		}
		processors[name] = processor
	}
	return processors
}

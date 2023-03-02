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

import (
	"context"
	"errors"
	"fmt"
	log "github.com/golang/glog"
	"github.com/intelligent-machine-learning/easydl/brain/pkg/common"
	"github.com/intelligent-machine-learning/easydl/brain/pkg/config/utils"
	"gopkg.in/yaml.v3"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/client-go/kubernetes"
	"sync"
	"time"
)

const (
	initCheckInterval = time.Second
	maxInitTimeout    = 5 * time.Minute
)

// This func will be called when updating config.
type configUpdateNotify func(newConfig *Config) error

// Manager is the struct of config manager
type Manager struct {
	sync.RWMutex

	controller    *Controller
	configmapKey  string
	configmapName string

	ctx             context.Context
	errReporter     common.ErrorReporter
	config          *Config
	configObservers map[string]configUpdateNotify
}

// NewManager create an instance of config map manager by the specified opts of config manager.
func NewManager(namespace string, configMapName string, configMapKey string, kubeClientSet kubernetes.Interface) *Manager {
	log.Infof("Create config manager with namespace: %s, configMapName: %s, configMapKey: %s", namespace, configMapName, configMapKey)
	return &Manager{
		controller:      NewController(namespace, configMapName, kubeClientSet),
		configmapKey:    configMapKey,
		configmapName:   configMapName,
		configObservers: make(map[string]configUpdateNotify),
	}
}

// RegisterConfigObserver registers a function which will be called when the config is updated
func (m *Manager) RegisterConfigObserver(observerName string, notify configUpdateNotify) {
	m.Lock()
	defer m.Unlock()

	if _, exist := m.configObservers[observerName]; exist {
		log.Infof("observer[%s] is existing. Overwrite it.", observerName)
	} else {
		log.Infof("observer[%s] is registered", observerName)
	}
	m.configObservers[observerName] = notify
}

// Run start to run the k8s configmap manager. It will start the configmap controller and try to
// get the initial config. After that, any changes during runtime will trigger the configmap
// controller to update the config.
func (m *Manager) Run(ctx context.Context, errReporter common.ErrorReporter) error {
	m.ctx = ctx
	m.errReporter = errReporter
	m.config = NewEmptyConfig()
	currConf, err := m.getConfigMap()
	if err != nil {
		log.Errorf("fail to get config %s: %v", m.configmapName, err)
		return err
	}
	log.Infof("currConf: %v", currConf)
	m.config.SetConfig(currConf)
	log.Infof("start with config: %v", m.config)

	m.controller.Run(ctx, func(newCM *corev1.ConfigMap) error {
		conf, err := parseConfigMap(newCM, m.configmapKey)
		if err != nil {
			return err
		}
		return m.UpdateConfig(NewConfig(conf))

	})

	// Wait until the initialization of config is done. If none configs was retrieved in a specified
	// MaxInitTimeout, an error will be raised to notified the master/executor server for the further
	// error handling.
	success := utils.WaitForCondition(
		func() bool { return m.config != nil && !m.config.IsEmpty() },
		initCheckInterval, maxInitTimeout)
	if !success {
		return errors.New("failed to get configuration when initialization")
	}
	return nil
}

// GetConfig returns the config
func (m *Manager) GetConfig() (*Config, error) {
	m.RLock()
	defer m.RUnlock()

	if m.config == nil {
		return nil, errors.New("initialization of config has NOT been finished yet")
	}
	return m.config.Clone(), nil
}

// UpdateConfig updates the config
func (m *Manager) UpdateConfig(newConfig *Config) error {
	if newConfig == nil {
		err := errors.New("the new config is NIL; skip the updating")
		log.Error(err)
		return err
	}

	m.Lock()
	defer m.Unlock()

	log.Infof("Start to update config.\nOld config: %v\n.New config: %v\n", m.config, newConfig)
	m.config.SetConfig(newConfig)

	cloneConf := m.config.Clone()
	for observer, notify := range m.configObservers {
		if err := notify(cloneConf); err != nil {
			log.Errorf("Failed to notify %s: %v", observer, err)
			m.errReporter.ReportError(m.ctx, common.NewError("ConfigManager-"+observer, err))
		}
		log.Infof("Notify %s successfully.", observer)
	}
	return nil
}

// Get the latest configuration from config map directly.
func (m *Manager) getConfigMap() (*Config, error) {
	configMap, err := m.controller.Get(m.ctx)
	if err != nil {
		return nil, err
	}
	conf, err := parseConfigMap(configMap, m.configmapKey)
	if err != nil {
		log.Errorf("Parse ConfigMap error: %v", err)
		return nil, err
	}
	log.Infof("get config: %v", conf)
	m.config.SetData(conf)
	return m.config.Clone(), nil
}

// Retrieve server configuration from ConfigMap by config key.
// The config retrieved from configmap will be parsed first and then converted to the Config.
func parseConfigMap(cm *corev1.ConfigMap, configKey string) (map[string]interface{}, error) {
	rawData, exist := cm.Data[configKey]
	if !exist {
		return nil, fmt.Errorf("not set config[key=%s] in ConfigMap[%v]", configKey, cm.Data)
	}

	// Content encoding is expected to be YAML.
	data := make(map[string]interface{})
	err := yaml.Unmarshal([]byte(rawData), data)
	if err != nil {
		return nil, err
	}
	return data, nil
}

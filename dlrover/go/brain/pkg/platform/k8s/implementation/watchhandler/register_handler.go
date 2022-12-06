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

package watchhandler

import (
	log "github.com/golang/glog"
	"github.com/intelligent-machine-learning/easydl/brain/pkg/config"
	datastoreapi "github.com/intelligent-machine-learning/easydl/brain/pkg/datastore/api"
	"github.com/intelligent-machine-learning/easydl/brain/pkg/platform/k8s/watcher/common"
	"sync"
)

const (
	workerConcurrent = 3
)

var (
	registerWatchHandlerFuncs = make(map[string]RegisterWatchHandlerFunc)
	registerFuncMutx          = &sync.RWMutex{}
)

// RegisterWatchHandlerFunc is the function to register a watch handler
type RegisterWatchHandlerFunc func(kubeWatcher *common.KubeWatcher, name string, conf *config.Config, dataStore datastoreapi.DataStore) error

func registerWatchHandlerFunc(name string, registerFunc RegisterWatchHandlerFunc) {
	registerFuncMutx.Lock()
	defer registerFuncMutx.Unlock()

	if _, found := registerWatchHandlerFuncs[name]; found {
		log.Errorf("Can not register RegisterWatchHandlerFunc func %s", name)
		return
	}
	registerWatchHandlerFuncs[name] = registerFunc
}

// GetRegisterWatchHandlerFuncs returns all registered watch handlers
func GetRegisterWatchHandlerFuncs() map[string]RegisterWatchHandlerFunc {
	registerFuncMutx.Lock()
	defer registerFuncMutx.Unlock()

	registerFuncs := make(map[string]RegisterWatchHandlerFunc)
	for name, registerFunc := range registerWatchHandlerFuncs {
		registerFuncs[name] = registerFunc
	}
	return registerFuncs
}

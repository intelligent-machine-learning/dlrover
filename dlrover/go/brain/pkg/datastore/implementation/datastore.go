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

package implementation

import (
	"fmt"
	log "github.com/golang/glog"
	"github.com/intelligent-machine-learning/easydl/brain/pkg/config"
	datastoreapi "github.com/intelligent-machine-learning/easydl/brain/pkg/datastore/api"
	"sync"
)

var (
	registerMutex = &sync.RWMutex{}
	newFuncs      = make(map[string]DataStoreNewFunc)
)

// DataStoreNewFunc is the new func of data stores
type DataStoreNewFunc func(conf *config.Config) (datastoreapi.DataStore, error)

func registerNewFunc(name string, newFunc DataStoreNewFunc) {
	registerMutex.Lock()
	defer registerMutex.Unlock()

	if _, exist := newFuncs[name]; exist {
		log.Errorf("DataStore new func %s has already registered", name)
		return
	}
	newFuncs[name] = newFunc
}

// CreateDataStore create a datastore for a given name
func CreateDataStore(name string, conf *config.Config) (datastoreapi.DataStore, error) {
	registerMutex.RLock()
	defer registerMutex.RUnlock()

	if _, exists := newFuncs[name]; !exists {
		err := fmt.Errorf("DataStore %s has not registered", name)
		return nil, err
	}

	dataStore, err := newFuncs[name](conf)
	if err != nil {
		return nil, err
	}

	return dataStore, nil
}

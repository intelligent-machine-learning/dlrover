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

package configretriever

import (
	"fmt"
	log "github.com/golang/glog"
	"github.com/intelligent-machine-learning/easydl/brain/pkg/config"
	"github.com/intelligent-machine-learning/easydl/brain/pkg/optimizer/api"
	"sync"
)

var (
	locker                  = &sync.RWMutex{}
	newConfigRetrieverFuncs = make(map[string]newConfigRetrieverFunc)
)

type newConfigRetrieverFunc func(config *config.Config) (api.ConfigRetriever, error)

func registerNewConfigRetrieverFunc(name string, newFunc newConfigRetrieverFunc) error {
	locker.Lock()
	defer locker.Unlock()

	if _, found := newConfigRetrieverFuncs[name]; found {
		err := fmt.Errorf("NewConfigRetrieverFunc %s has already registered", name)
		return err
	}
	newConfigRetrieverFuncs[name] = newFunc
	return nil
}

// CreateConfigRetriever creates a specified config retriever
func CreateConfigRetriever(name string, conf *config.Config) (api.ConfigRetriever, error) {
	locker.Lock()
	defer locker.Unlock()

	newFunc, exist := newConfigRetrieverFuncs[name]
	if !exist {
		err := fmt.Errorf("Config retriever %s has not registered", name)
		return nil, err
	}
	retriever, err := newFunc(conf)
	if err != nil {
		log.Errorf("fail to create config retriever %s: %v", name, err)
		return nil, err
	}
	return retriever, nil
}

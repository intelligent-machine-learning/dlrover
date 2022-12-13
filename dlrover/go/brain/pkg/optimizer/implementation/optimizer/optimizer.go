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
	"fmt"
	"github.com/intelligent-machine-learning/easydl/brain/pkg/config"
	datastoreapi "github.com/intelligent-machine-learning/easydl/brain/pkg/datastore/api"
	"github.com/intelligent-machine-learning/easydl/brain/pkg/optimizer/api"
	"sync"
)

const (
	defaultStepCountThreshold = 5
)

var (
	locker            = &sync.RWMutex{}
	optimizers        = make(map[string]api.Optimizer)
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

// CreateOptimizer create a specified optimizer
func CreateOptimizer(name string, dataStore datastoreapi.DataStore, config *config.Config) (api.Optimizer, error) {
	locker.Lock()
	defer locker.Unlock()

	newFunc, exist := newOptimizerFuncs[name]
	if !exist {
		err := fmt.Errorf("Optimizer %s has not registered", name)
		return nil, err
	}
	optimizer := newFunc(dataStore, config)
	return optimizer, nil
}

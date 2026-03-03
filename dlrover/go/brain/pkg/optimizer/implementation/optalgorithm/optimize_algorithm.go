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

package optalgorithm

import (
	"fmt"
	"github.com/intelligent-machine-learning/easydl/brain/pkg/common"
	datastoreapi "github.com/intelligent-machine-learning/easydl/brain/pkg/datastore/api"
	optconfig "github.com/intelligent-machine-learning/easydl/brain/pkg/optimizer/config"
	"sync"
)

const (
	initStepTime      = 1800
	defaultInitWorker = 10
)

var (
	locker        = &sync.RWMutex{}
	optAlgorithms = make(map[string]optimizeAlgorithm)
)

type optimizeAlgorithm func(dataStore datastoreapi.DataStore, conf *optconfig.OptimizeAlgorithmConfig, job *common.OptimizeJobMeta,
	jobs []*common.OptimizeJobMeta) (*common.AlgorithmOptimizePlan, error)

func registerOptimizeAlgorithm(name string, optAlgorithm optimizeAlgorithm) error {
	locker.Lock()
	defer locker.Unlock()

	if _, found := optAlgorithms[name]; found {
		err := fmt.Errorf("%s optimize algorithm has already registered", name)
		return err
	}

	optAlgorithms[name] = optAlgorithm
	return nil
}

// Optimize calls responding resource optimizer function to generate resource optimize plan
func Optimize(dataStore datastoreapi.DataStore, conf *optconfig.OptimizeAlgorithmConfig, job *common.OptimizeJobMeta,
	jobs []*common.OptimizeJobMeta) (*common.AlgorithmOptimizePlan, error) {
	if conf == nil {
		err := fmt.Errorf("resource optimize config is nil")
		return nil, err
	}
	locker.RLock()
	f := optAlgorithms[conf.Name]
	locker.RUnlock()

	if f == nil {
		err := fmt.Errorf("%s resource optimize func does not register", conf.Name)
		return nil, err
	}
	return f(dataStore, conf, job, jobs)
}

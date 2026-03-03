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
	log "github.com/golang/glog"
	"github.com/intelligent-machine-learning/easydl/brain/pkg/common"
	"github.com/intelligent-machine-learning/easydl/brain/pkg/config"
	datastoreapi "github.com/intelligent-machine-learning/easydl/brain/pkg/datastore/api"
	optapi "github.com/intelligent-machine-learning/easydl/brain/pkg/optimizer/api"
	optconfig "github.com/intelligent-machine-learning/easydl/brain/pkg/optimizer/config"
)

const (
	// BaseOptimizerName is the name of BaseOptimizer
	BaseOptimizerName = "base_optimizer"
)

// BaseOptimizer is the basic optimizer
type BaseOptimizer struct {
	dataStore datastoreapi.DataStore
	config    *config.Config
}

func init() {
	registerNewOptimizerFunc(BaseOptimizerName, newBaseOptimizer)
}

func newBaseOptimizer(dataStore datastoreapi.DataStore, config *config.Config) optapi.Optimizer {
	return &BaseOptimizer{
		dataStore: dataStore,
		config:    config,
	}
}

// Optimize optimizes the chief worker initial resources
func (optimizer *BaseOptimizer) Optimize(conf *optconfig.OptimizerConfig, jobs []*common.JobMeta) ([]*common.OptimizePlan, error) {
	log.Infof("base optimizer current config: %v", optimizer.config)
	return nil, nil
}

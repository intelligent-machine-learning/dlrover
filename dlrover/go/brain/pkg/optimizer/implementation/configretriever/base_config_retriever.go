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
	"github.com/intelligent-machine-learning/easydl/brain/pkg/config"
	optapi "github.com/intelligent-machine-learning/easydl/brain/pkg/optimizer/api"
	optconfig "github.com/intelligent-machine-learning/easydl/brain/pkg/optimizer/config"
	pb "github.com/intelligent-machine-learning/easydl/brain/pkg/proto"
)

const (
	baseConfigRetrieverName = "base_config_retriever"
)

type baseConfigRetriever struct {
	conf *config.Config
}

func init() {
	registerNewConfigRetrieverFunc(baseConfigRetrieverName, newBaseConfigRetriever)
}

func newBaseConfigRetriever(conf *config.Config) (optapi.ConfigRetriever, error) {
	return &baseConfigRetriever{
		conf: conf,
	}, nil
}

// Retrieve retrieve optimizer config from pb optimize config
func (r *baseConfigRetriever) Retrieve(c *pb.OptimizeConfig) (*optconfig.OptimizerConfig, error) {
	if c == nil {
		err := fmt.Errorf("the config is nil")
		return nil, err
	}

	if c.CustomizedConfig == nil {
		return &optconfig.OptimizerConfig{
			OptimizeAlgorithmConfig: &optconfig.OptimizeAlgorithmConfig{},
		}, nil
	}

	return &optconfig.OptimizerConfig{
		Name: c.CustomizedConfig[optconfig.ConfigRetrieverConfigKeyOptimizerName],
		OptimizeAlgorithmConfig: &optconfig.OptimizeAlgorithmConfig{
			Name:             c.CustomizedConfig[optconfig.ConfigRetrieverConfigKeyOptimizeAlgorithm],
			CustomizedConfig: c.CustomizedConfig,
		},
	}, nil
}

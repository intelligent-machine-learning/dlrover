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

const (
	// ConfigRetrieverConfigKeyOptimizerName is the config key of optimization name in config retriever
	ConfigRetrieverConfigKeyOptimizerName = "optimization"
	// ConfigRetrieverConfigKeyOptimizeAlgorithm is the config key of optimize algorithm in config retriever
	ConfigRetrieverConfigKeyOptimizeAlgorithm = "optimize-algorithm"
)

// OptimizeAlgorithmConfig is the config of optimize algorithm
type OptimizeAlgorithmConfig struct {
	// Name is the name of optimize config
	Name string
	// CustomizedConfig is the customized configure
	CustomizedConfig map[string]string
}

// OptimizerConfig is the config of optimization
type OptimizerConfig struct {
	// Name is the name of optimization
	Name string
	// CustomizedConfig is the customized configure of the optimization
	CustomizedConfig map[string]string
	// ResOptFuncConfig is the config of resource optimize function
	OptimizeAlgorithmConfig *OptimizeAlgorithmConfig
}

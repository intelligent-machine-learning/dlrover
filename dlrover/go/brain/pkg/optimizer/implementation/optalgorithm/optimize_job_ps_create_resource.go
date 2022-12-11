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
	"github.com/intelligent-machine-learning/easydl/brain/pkg/common"
	datastoreapi "github.com/intelligent-machine-learning/easydl/brain/pkg/datastore/api"
	optconfig "github.com/intelligent-machine-learning/easydl/brain/pkg/optimizer/config"
	optimplutils "github.com/intelligent-machine-learning/easydl/brain/pkg/optimizer/implementation/utils"
)

const (
	// OptimizeAlgorithmJobPSCreateResource is the name of job ps create resource optimize function
	OptimizeAlgorithmJobPSCreateResource = "optimize_job_ps_create_resource"
)

func init() {
	registerOptimizeAlgorithm(OptimizeAlgorithmJobPSCreateResource, OptimizeJobPSCreateResource)
}

// OptimizeJobPSCreateResource optimizes job ps initial resources
func OptimizeJobPSCreateResource(dataStore datastoreapi.DataStore, conf *optconfig.OptimizeAlgorithmConfig, optJob *common.OptimizeJobMeta,
	historyJobs []*common.OptimizeJobMeta) (*common.AlgorithmOptimizePlan, error) {
	return optimplutils.EstimateJobResourceByHistoricJobs(conf, optJob.Metrics, historyJobs)
}

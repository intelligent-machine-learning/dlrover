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
	"github.com/intelligent-machine-learning/easydl/brain/pkg/config"
	optconfig "github.com/intelligent-machine-learning/easydl/brain/pkg/optimizer/config"
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestOptimizeJobPSColdCreateResource(t *testing.T) {
	conf := &optconfig.OptimizeAlgorithmConfig{
		CustomizedConfig: map[string]string{
			config.OptimizerPSColdReplica: "4",
			config.OptimizerPSColdCPU:     "12",
			config.OptimizerPSColdMemory:  "1000",
		},
	}

	plan, err := OptimizeJobPSColdCreateResource(nil, conf, nil, nil)
	assert.NoError(t, err)
	assert.NotEqual(t, plan, nil)

	res := plan.JobRes.TaskGroupResources[common.PSTaskGroupName]
	assert.Equal(t, res.Count, int32(4))
	assert.Equal(t, res.Resource.Memory, 1000.0)
	assert.Equal(t, res.Resource.CPUCore, float32(12))
}

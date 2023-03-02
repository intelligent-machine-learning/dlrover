// Copyright 2023 The DLRover Authors. All rights reserved.
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

package common

import (
	"github.com/intelligent-machine-learning/easydl/brain/pkg/common"
	"github.com/intelligent-machine-learning/easydl/brain/pkg/optimizer/config"
)

// OptimizeEvent is the struct of optimize event
type OptimizeEvent struct {
	Type           string
	ProcessorName  string
	DataStoreName  string
	Jobs           []*common.JobMeta
	Conf           *config.OptimizerConfig
	CustomizedData map[string]string
}

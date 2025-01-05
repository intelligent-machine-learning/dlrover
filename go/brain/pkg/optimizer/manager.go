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
	"context"
	"github.com/intelligent-machine-learning/easydl/brain/pkg/common"
	"github.com/intelligent-machine-learning/easydl/brain/pkg/config"
	"github.com/intelligent-machine-learning/easydl/brain/pkg/optimizer/implementation/optprocessor"
	pb "github.com/intelligent-machine-learning/easydl/brain/pkg/proto"
)

// Manager is the struct to handle optimize request
type Manager struct {
	processorManager *optprocessor.Manager
}

// NewManager creates a new OptimizerManager
func NewManager(conf *config.Config) *Manager {
	processorManager := optprocessor.NewManager(conf)

	return &Manager{
		processorManager: processorManager,
	}
}

// Run starts the manager
func (m *Manager) Run(ctx context.Context, errReporter common.ErrorReporter) error {
	return m.processorManager.Run(ctx, errReporter)
}

// ProcessOptimizeRequest processes optimize request
func (m *Manager) ProcessOptimizeRequest(ctx context.Context, request *pb.OptimizeRequest) ([]*pb.JobOptimizePlan, error) {
	return m.processorManager.ProcessOptimizeRequest(ctx, request)
}

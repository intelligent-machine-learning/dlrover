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

package optprocessor

import (
	"context"
	"fmt"
	log "github.com/golang/glog"
	"github.com/intelligent-machine-learning/easydl/brain/pkg/common"
	"github.com/intelligent-machine-learning/easydl/brain/pkg/config"
	"github.com/intelligent-machine-learning/easydl/brain/pkg/datastore"
	optimizerapi "github.com/intelligent-machine-learning/easydl/brain/pkg/optimizer/api"
	optcommon "github.com/intelligent-machine-learning/easydl/brain/pkg/optimizer/common"
	"github.com/intelligent-machine-learning/easydl/brain/pkg/optimizer/implementation/optalgorithm"
	imploptimizer "github.com/intelligent-machine-learning/easydl/brain/pkg/optimizer/implementation/optimizer"
)

const (
	// RunningTrainingJobOptimizeRequestProcessorName is the name of optimize request processor which optimizes running jobs
	RunningTrainingJobOptimizeRequestProcessorName = "running_training_job_optimize_request_processor"
)

func init() {
	registerNewOptimizeRequestOptimizerFunc(RunningTrainingJobOptimizeRequestProcessorName, newRunningTrainingJobOptimizeRequestProcessor)
}

// RunningTrainingJobOptimizeRequestProcessor is the struct of running job optimize request processor
type RunningTrainingJobOptimizeRequestProcessor struct {
	optimizerManager *imploptimizer.Manager
	conf             *config.Config
	dsManager        *datastore.Manager
}

func newRunningTrainingJobOptimizeRequestProcessor(conf *config.Config, dsManager *datastore.Manager,
	optimizerManager *imploptimizer.Manager) optimizerapi.OptimizeRequestProcessor {

	return &RunningTrainingJobOptimizeRequestProcessor{
		optimizerManager: optimizerManager,
		conf:             conf,
		dsManager:        dsManager,
	}
}

// Optimize process the optimize event synchronously
func (p *RunningTrainingJobOptimizeRequestProcessor) Optimize(ctx context.Context, event *optcommon.OptimizeEvent) ([]*common.OptimizePlan, error) {
	var plan *common.OptimizePlan
	var err error

	if event.Type == optcommon.JobStageCreate {
		plan, err = p.optimizeCreateJob(ctx, event)
	} else if event.Type == optcommon.JobStagePSInitial {
		plan, err = p.optimizeInitialJobPS(ctx, event)
	} else if event.Type == optcommon.JobStageWorkerInitial {
		plan, err = p.optimizeInitialJobWorker(ctx, event)
	} else if event.Type == optcommon.JobStageRunning {
		plan, err = p.optimizeRunningJob(ctx, event)
	} else {
		err = fmt.Errorf("invalid event type %s", event.Type)
		plan = nil
	}

	plans := make([]*common.OptimizePlan, 0)
	if plan != nil {
		plan.JobMeta = event.Jobs[0]
		plans = append(plans, plan)
	}

	return plans, err
}

func (p *RunningTrainingJobOptimizeRequestProcessor) optimizeCreateJob(ctx context.Context, event *optcommon.OptimizeEvent) (*common.OptimizePlan, error) {
	event.Conf.Name = imploptimizer.JobWorkerCreateResourceOptimizerName
	event.Conf.OptimizeAlgorithmConfig.Name = optalgorithm.OptimizeAlgorithmJobWorkerCreateResource
	chiefWorkerPlan, err := p.optimizerManager.Optimize(event.Conf, event.Jobs)
	if err != nil {
		log.Errorf("fail to optimize chief worker: %v", err)
	} else if len(chiefWorkerPlan) == 0 {
		log.Errorf("fail to optimize chief worker")
	}

	event.Conf.Name = imploptimizer.JobPSCreateResourceOptimizerName
	psPlan, err := p.optimizerManager.Optimize(event.Conf, event.Jobs)
	if err != nil {
		log.Errorf("fail to optimize create ps: %v", err)
	} else if len(psPlan) == 0 {
		log.Errorf("fail to optimize create ps")
	}

	plan := &common.OptimizePlan{
		AlgOptPlan: &common.AlgorithmOptimizePlan{
			JobRes: &common.JobResource{
				TaskGroupResources: make(map[string]*common.TaskGroupResource),
			},
		},
	}

	if len(chiefWorkerPlan) > 0 {
		if tgRes := getTaskGroupResource(chiefWorkerPlan[0], common.WorkerTaskGroupName); tgRes != nil {
			plan.AlgOptPlan.JobRes.TaskGroupResources[common.WorkerTaskGroupName] = tgRes
		}
	}

	if len(psPlan) > 0 {
		if tgRes := getTaskGroupResource(psPlan[0], common.PSTaskGroupName); tgRes != nil {
			plan.AlgOptPlan.JobRes.TaskGroupResources[common.PSTaskGroupName] = tgRes
		}
	}

	return plan, nil
}

func (p *RunningTrainingJobOptimizeRequestProcessor) optimizeInitialJobPS(ctx context.Context, event *optcommon.OptimizeEvent) (*common.OptimizePlan, error) {
	event.Conf.Name = imploptimizer.JobPSInitAdjustResourceOptimizerName
	event.Conf.OptimizeAlgorithmConfig.Name = optalgorithm.OptimizeAlgorithmJobPSInitAdjustResource
	psPlan, err := p.optimizerManager.Optimize(event.Conf, event.Jobs)
	if err != nil {
		log.Errorf("fail to process event %v: %v", event, err)
		return nil, err
	} else if len(psPlan) == 0 {
		err = fmt.Errorf("fail to process event %v", event)
		return nil, err
	}
	plan := &common.OptimizePlan{
		AlgOptPlan: &common.AlgorithmOptimizePlan{
			JobRes: &common.JobResource{
				TaskGroupResources: make(map[string]*common.TaskGroupResource),
			},
		},
	}
	if tgRes := getTaskGroupResource(psPlan[0], common.PSTaskGroupName); tgRes != nil {
		plan.AlgOptPlan.JobRes.TaskGroupResources[common.PSTaskGroupName] = tgRes
	}
	return plan, nil
}

func (p *RunningTrainingJobOptimizeRequestProcessor) optimizeInitialJobWorker(ctx context.Context, event *optcommon.OptimizeEvent) (*common.OptimizePlan, error) {
	event.Conf.Name = imploptimizer.JobWorkerResourceOptimizerName
	event.Conf.OptimizeAlgorithmConfig.Name = optalgorithm.OptimizeAlgorithmJobWorkerResource
	workerPlan, err := p.optimizerManager.Optimize(event.Conf, event.Jobs)
	if err != nil {
		log.Errorf("fail to optimize worker: %v", err)
		return nil, err
	}
	if len(workerPlan) == 0 {
		err = fmt.Errorf("fail to optimize worker")
		return nil, err
	}

	return workerPlan[0], nil
}

func (p *RunningTrainingJobOptimizeRequestProcessor) optimizeRunningJob(ctx context.Context, event *optcommon.OptimizeEvent) (*common.OptimizePlan, error) {
	event.Conf.Name = imploptimizer.JobPSRunningResourceOptimizerName
	psPlan, err := p.optimizerManager.Optimize(event.Conf, event.Jobs)
	if err != nil {
		log.Errorf("fail to optimize running job ps: %v", err)
		return nil, err
	}
	if len(psPlan) == 0 {
		err = fmt.Errorf("fail to optimize running job ps")
		return nil, err
	}

	return psPlan[0], nil
}

func getTaskGroupResource(plan *common.OptimizePlan, name string) *common.TaskGroupResource {
	if plan == nil || plan.AlgOptPlan == nil || plan.AlgOptPlan.JobRes == nil || plan.AlgOptPlan.JobRes.TaskGroupResources == nil {
		return nil
	}
	return plan.AlgOptPlan.JobRes.TaskGroupResources[name]
}

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
	"encoding/json"
	"fmt"
	log "github.com/golang/glog"
	"github.com/intelligent-machine-learning/easydl/brain/pkg/common"
	"github.com/intelligent-machine-learning/easydl/brain/pkg/config"
	datastoreapi "github.com/intelligent-machine-learning/easydl/brain/pkg/datastore/api"
	"github.com/intelligent-machine-learning/easydl/brain/pkg/datastore/recorder/mysql"
	optimizerapi "github.com/intelligent-machine-learning/easydl/brain/pkg/optimizer/api"
	optconfig "github.com/intelligent-machine-learning/easydl/brain/pkg/optimizer/config"
	optimplcomm "github.com/intelligent-machine-learning/easydl/brain/pkg/optimizer/implementation/common"
	"github.com/intelligent-machine-learning/easydl/brain/pkg/optimizer/implementation/optalgorithm"
	"github.com/intelligent-machine-learning/easydl/brain/pkg/utils"
	"strconv"
)

const (
	// JobPSInitAdjustResourceOptimizerName is the name of optimizer which optimize jpb ps resources when it is just running
	JobPSInitAdjustResourceOptimizerName = "job_ps_init_adjust_resource_optimizer"

	defaultPSCPUMargin                               = 4
	defaultOptimizerPSMemoryWorkloadUnbalancePercent = 0.2
	defaultOptimizerTargetWorkerCount                = 20
)

// JobPSInitAdjustResourceOptimizer is the optimizer which optimizes job ps resources when it is just running
type JobPSInitAdjustResourceOptimizer struct {
	dataStore datastoreapi.DataStore
	config    *config.Config
}

func init() {
	registerNewOptimizerFunc(JobPSInitAdjustResourceOptimizerName, newJobPSInitAdjustResourceOptimizer)
}

func newJobPSInitAdjustResourceOptimizer(dataStore datastoreapi.DataStore, config *config.Config) optimizerapi.Optimizer {
	return &JobPSInitAdjustResourceOptimizer{
		dataStore: dataStore,
		config:    config,
	}
}

// Optimize optimizes the chief worker initial resources
func (optimizer *JobPSInitAdjustResourceOptimizer) Optimize(conf *optconfig.OptimizerConfig, jobMetas []*common.JobMeta) ([]*common.OptimizePlan, error) {
	if conf == nil || conf.OptimizeAlgorithmConfig == nil {
		err := fmt.Errorf("invalid optimizer config: %v", conf)
		return nil, err
	}
	if len(jobMetas) == 0 {
		return nil, nil
	}

	jobMeta := jobMetas[0]

	cond := &datastoreapi.Condition{
		Type: common.TypeGetDataGetJobMetrics,
		Extra: &mysql.JobMetricsCondition{
			UID: jobMeta.UUID,
		},
	}
	jobMetrics := &mysql.JobMetrics{}
	err := optimizer.dataStore.GetData(cond, jobMetrics)
	if err != nil {
		log.Errorf("fail to get job metrics for %s: %v", jobMeta.UUID, err)
		return nil, err
	}

	cond = &datastoreapi.Condition{
		Type: common.TypeGetDataGetJob,
		Extra: &mysql.JobCondition{
			UID: jobMeta.UUID,
		},
	}
	job := &mysql.Job{}
	err = optimizer.dataStore.GetData(cond, job)
	if err != nil {
		log.Errorf("fail to get job  for %s: %v", jobMeta.UUID, err)
		return nil, err
	}

	optJob := &common.OptimizeJobMeta{
		JobMeta: jobMeta,
		Metrics: utils.ConvertDBJobMetricsToJobMetrics(jobMetrics),
	}

	if conf.OptimizeAlgorithmConfig.CustomizedConfig == nil {
		conf.OptimizeAlgorithmConfig.CustomizedConfig = make(map[string]string)
	}

	isOom := false
	jobStatus := &common.JobStatus{}
	err = json.Unmarshal([]byte(job.Status), jobStatus)
	if err != nil {
		log.Errorf("fail to unmarshal job status %s for %s: %v", job.Status, job.Name, err)
	} else if jobStatus.IsOOM {
		isOom = true
	}

	if isOom {
		workloadUnbalancePercent := optimizer.config.GetFloat64WithValue(config.OptimizerPSMemoryWorkloadUnbalancePercent, defaultOptimizerPSMemoryWorkloadUnbalancePercent)
		conf.OptimizeAlgorithmConfig.CustomizedConfig[config.OptimizerPSMemoryWorkloadUnbalancePercent] = fmt.Sprintf("%f", workloadUnbalancePercent)

		conf.OptimizeAlgorithmConfig.Name = optalgorithm.OptimizeAlgorithmJobPSOomResource

		resOptPlan, err := optalgorithm.Optimize(optimizer.dataStore, conf.OptimizeAlgorithmConfig, optJob, nil)

		if err != nil {
			return nil, err
		}
		if resOptPlan == nil {
			return nil, nil
		}
		plan := &common.OptimizePlan{
			JobMeta:    jobMeta,
			AlgOptPlan: resOptPlan,
		}
		return []*common.OptimizePlan{plan}, nil
	}

	psMarginCPU := optimizer.config.GetIntWithValue(config.OptimizerPSMarginCPU, defaultPSCPUMargin)
	conf.OptimizeAlgorithmConfig.CustomizedConfig[config.OptimizerPSMarginCPU] = strconv.Itoa(psMarginCPU)

	psMarginMemoryPercent := optimizer.config.GetFloat64WithValue(config.OptimizerPSMemoryMarginPercent, optimplcomm.DefaultPSMemoryMarginPercent)
	conf.OptimizeAlgorithmConfig.CustomizedConfig[config.OptimizerPSMemoryMarginPercent] = fmt.Sprintf("%f", psMarginMemoryPercent)

	stepCountThreshold := optimizer.config.GetIntWithValue(config.OptimizerStepCountThreshold, defaultStepCountThreshold)
	conf.OptimizeAlgorithmConfig.CustomizedConfig[config.OptimizerStepCountThreshold] = strconv.Itoa(stepCountThreshold)

	targetWorkerCount := optimizer.config.GetIntWithValue(config.OptimizerPSInitAdjustTargetWorkerCount, defaultOptimizerTargetWorkerCount)
	conf.OptimizeAlgorithmConfig.CustomizedConfig[config.OptimizerPSInitAdjustTargetWorkerCount] = strconv.Itoa(targetWorkerCount)

	conf.OptimizeAlgorithmConfig.Name = optalgorithm.OptimizeAlgorithmJobPSInitAdjustResource

	resOptPlan, err := optalgorithm.Optimize(optimizer.dataStore, conf.OptimizeAlgorithmConfig, optJob, nil)

	if err != nil {
		return nil, err
	}
	if resOptPlan == nil {
		return nil, nil
	}
	plan := &common.OptimizePlan{
		JobMeta:    jobMeta,
		AlgOptPlan: resOptPlan,
	}
	return []*common.OptimizePlan{plan}, nil
}

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
	"fmt"
	log "github.com/golang/glog"
	"github.com/intelligent-machine-learning/easydl/brain/pkg/common"
	"github.com/intelligent-machine-learning/easydl/brain/pkg/config"
	datastoreapi "github.com/intelligent-machine-learning/easydl/brain/pkg/datastore/api"
	"github.com/intelligent-machine-learning/easydl/brain/pkg/datastore/recorder/mysql"
	optimizerapi "github.com/intelligent-machine-learning/easydl/brain/pkg/optimizer/api"
	optconfig "github.com/intelligent-machine-learning/easydl/brain/pkg/optimizer/config"
	"github.com/intelligent-machine-learning/easydl/brain/pkg/optimizer/implementation/optalgorithm"
	"github.com/intelligent-machine-learning/easydl/brain/pkg/utils"
	"strconv"
)

const (
	// JobWorkerResourceOptimizerName is the name of JobWorkerResourceOptimizer
	JobWorkerResourceOptimizerName    = "job_worker_resource_optimizer"
	defaultTrainingSpeedLessPercent   = 0.1
	defaultWorkerReplicaDecreaseCount = 4
	defaultPSExhaustedThreshold       = 0.95
	defaultWorkerMaxInitCountPerStep  = 32
	defaultWorkerMaxCountPerStep      = 4
	defaultWorkerMemoryMarginPercent  = 0.2
	defaultWorkerCPUMarginCore        = 1
	defaultWorkerCPUUtilLessPercent   = 0.15
	defaultWorkerCPUUtilCompCount     = 2
	defaultWorkerMaxReplicaCount      = 60
)

// JobWorkerResourceOptimizer is the optimizer which optimizes worker resources at runtime.
type JobWorkerResourceOptimizer struct {
	dataStore datastoreapi.DataStore
	config    *config.Config
}

func init() {
	registerNewOptimizerFunc(JobWorkerResourceOptimizerName, newJobWorkerResourceOptimizer)
}

func newJobWorkerResourceOptimizer(dataStore datastoreapi.DataStore, config *config.Config) optimizerapi.Optimizer {
	return &JobWorkerResourceOptimizer{
		dataStore: dataStore,
		config:    config,
	}
}

// Optimize optimizes the worker resource at runtime
func (optimizer *JobWorkerResourceOptimizer) Optimize(conf *optconfig.OptimizerConfig, jobs []*common.JobMeta) ([]*common.OptimizePlan, error) {
	if conf == nil || conf.OptimizeAlgorithmConfig == nil {
		err := fmt.Errorf("invalid optimizer config: %v", conf)
		return nil, err
	}
	if len(jobs) == 0 {
		return nil, nil
	}

	job := jobs[0]
	// get the user of the job
	cond := &datastoreapi.Condition{
		Type: common.TypeGetDataGetJobMetrics,
		Extra: &mysql.JobMetricsCondition{
			UID: job.UUID,
		},
	}
	jobMetrics := &mysql.JobMetrics{}
	err := optimizer.dataStore.GetData(cond, jobMetrics)
	if err != nil {
		log.Errorf("Fail to get job_metrics for %s: %v", job.UUID, err)
		return nil, err
	}

	optJob := &common.OptimizeJobMeta{
		JobMeta: job,
		Metrics: utils.ConvertDBJobMetricsToJobMetrics(jobMetrics),
	}

	if conf.OptimizeAlgorithmConfig.CustomizedConfig == nil {
		conf.OptimizeAlgorithmConfig.CustomizedConfig = make(map[string]string)
	}

	maxReplicaCount := optimizer.config.GetIntWithValue(config.OptimizerWorkerMaxReplicaCount, defaultWorkerMaxReplicaCount)
	conf.OptimizeAlgorithmConfig.CustomizedConfig[config.OptimizerWorkerMaxReplicaCount] = strconv.Itoa(maxReplicaCount)

	stepCountThreshold := optimizer.config.GetIntWithValue(config.OptimizerStepCountThreshold, defaultStepCountThreshold)
	conf.OptimizeAlgorithmConfig.CustomizedConfig[config.OptimizerStepCountThreshold] = strconv.Itoa(stepCountThreshold)

	speedLessPercent := optimizer.config.GetFloat64WithValue(conf.CustomizedConfig[config.OptimizerTrainingSpeedLessPercent], defaultTrainingSpeedLessPercent)
	conf.OptimizeAlgorithmConfig.CustomizedConfig[config.OptimizerTrainingSpeedLessPercent] = fmt.Sprintf("%f", speedLessPercent)

	workerReplicaDecreaseCount := optimizer.config.GetIntWithValue(config.OptimizerWorkerReplicaDecreaseCount, defaultWorkerReplicaDecreaseCount)
	conf.OptimizeAlgorithmConfig.CustomizedConfig[config.OptimizerWorkerReplicaDecreaseCount] = strconv.Itoa(workerReplicaDecreaseCount)

	psCPUOverload := optimizer.config.GetFloat64WithValue(config.OptimizerPSCPUOverload, defaultPSCPUOverload)
	conf.OptimizeAlgorithmConfig.CustomizedConfig[config.OptimizerPSCPUOverload] = fmt.Sprintf("%f", psCPUOverload)

	psCPUExhaustedThreshold := optimizer.config.GetFloat64WithValue(config.OptimizerPSCPUExhaustedThreshold, defaultPSExhaustedThreshold)
	conf.OptimizeAlgorithmConfig.CustomizedConfig[config.OptimizerPSCPUExhaustedThreshold] = fmt.Sprintf("%f", psCPUExhaustedThreshold)

	workerMaxInitCountPerStep := optimizer.config.GetIntWithValue(config.OptimizerWorkerMaxInitCountPerStep, defaultWorkerMaxInitCountPerStep)
	conf.OptimizeAlgorithmConfig.CustomizedConfig[config.OptimizerWorkerMaxInitCountPerStep] = strconv.Itoa(workerMaxInitCountPerStep)

	workerMaxCountPerStep := optimizer.config.GetIntWithValue(config.OptimizerWorkerMaxCountPerStep, defaultWorkerMaxCountPerStep)
	conf.OptimizeAlgorithmConfig.CustomizedConfig[config.OptimizerWorkerMaxCountPerStep] = strconv.Itoa(workerMaxCountPerStep)

	workerMemoryMarginPercent := optimizer.config.GetFloat64WithValue(config.OptimizerWorkerMemoryMarginPercent, defaultWorkerMemoryMarginPercent)
	conf.OptimizeAlgorithmConfig.CustomizedConfig[config.OptimizerWorkerMemoryMarginPercent] = fmt.Sprintf("%f", workerMemoryMarginPercent)

	workerCPUMarginCores := optimizer.config.GetIntWithValue(config.OptimizerWorkerCPUMarginCore, defaultWorkerCPUMarginCore)
	conf.OptimizeAlgorithmConfig.CustomizedConfig[config.OptimizerWorkerCPUMarginCore] = strconv.Itoa(workerCPUMarginCores)

	workerCPUUtilCompCount := optimizer.config.GetIntWithValue(config.OptimizerWorkerCPUUtilCompCount, defaultWorkerCPUUtilCompCount)
	conf.OptimizeAlgorithmConfig.CustomizedConfig[config.OptimizerWorkerCPUUtilCompCount] = strconv.Itoa(workerCPUUtilCompCount)

	workerCPUUtilLessPercent := optimizer.config.GetFloat64WithValue(config.OptimizerWorkerCPUUtilLessPercent, defaultWorkerCPUUtilLessPercent)
	conf.OptimizeAlgorithmConfig.CustomizedConfig[config.OptimizerWorkerCPUUtilLessPercent] = fmt.Sprintf("%f", workerCPUUtilLessPercent)

	algOptPlan, err := optalgorithm.Optimize(optimizer.dataStore, conf.OptimizeAlgorithmConfig, optJob, nil)
	if err != nil {
		return nil, err
	}
	if algOptPlan == nil {
		return nil, nil
	}
	plan := &common.OptimizePlan{
		JobMeta:    job,
		AlgOptPlan: algOptPlan,
	}
	return []*common.OptimizePlan{plan}, nil
}

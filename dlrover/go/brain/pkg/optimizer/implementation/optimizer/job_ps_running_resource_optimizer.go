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
	optapi "github.com/intelligent-machine-learning/easydl/brain/pkg/optimizer/api"
	optconfig "github.com/intelligent-machine-learning/easydl/brain/pkg/optimizer/config"
	optimplcomm "github.com/intelligent-machine-learning/easydl/brain/pkg/optimizer/implementation/common"
	"github.com/intelligent-machine-learning/easydl/brain/pkg/optimizer/implementation/optalgorithm"
	"github.com/intelligent-machine-learning/easydl/brain/pkg/utils"
	"strconv"
)

const (
	// JobPSRunningResourceOptimizerName is the name of JobPSRunningResourceOptimizer
	JobPSRunningResourceOptimizerName = "job_ps_running_resource_optimizer"

	defaultHotPSCPUThreshold         = 0.8
	defaultHotPSMemoryThreshold      = 0.9
	defaultHotPSCPUTargetWorkerCount = 32
	defaultHotPSMemoryAdjust         = 8589934592 // 8GB
	defaultLowCPUThreshold           = 0.4
)

// JobPSRunningResourceOptimizer is the optimizer which optimizes job ps create resources
type JobPSRunningResourceOptimizer struct {
	dataStore datastoreapi.DataStore
	config    *config.Config
}

func init() {
	registerNewOptimizerFunc(JobPSRunningResourceOptimizerName, newJobPSRunningResourceOptimizer)
}

func newJobPSRunningResourceOptimizer(dataStore datastoreapi.DataStore, config *config.Config) optapi.Optimizer {
	return &JobPSRunningResourceOptimizer{
		dataStore: dataStore,
		config:    config,
	}
}

// Optimize optimizes the chief worker initial resources
func (optimizer *JobPSRunningResourceOptimizer) Optimize(conf *optconfig.OptimizerConfig, jobMetas []*common.JobMeta) ([]*common.OptimizePlan, error) {
	if conf == nil || conf.OptimizeAlgorithmConfig == nil {
		err := fmt.Errorf("invalid optimizer config: %v", conf)
		return nil, err
	}
	if len(jobMetas) == 0 {
		return nil, nil
	}

	jobMeta := jobMetas[0]
	// get the user of the job
	cond := &datastoreapi.Condition{
		Type: common.TypeGetDataGetJobMetrics,
		Extra: &mysql.JobMetricsCondition{
			UID: jobMeta.UUID,
		},
	}
	jobMetrics := &mysql.JobMetrics{}
	err := optimizer.dataStore.GetData(cond, jobMetrics)
	if err != nil {
		log.Errorf("Fail to get job_metrics for %s: %v", jobMeta.UUID, err)
		return nil, err
	}

	optJob := &common.OptimizeJobMeta{
		JobMeta: jobMeta,
		Metrics: utils.ConvertDBJobMetricsToJobMetrics(jobMetrics),
	}

	if conf.OptimizeAlgorithmConfig.CustomizedConfig == nil {
		conf.OptimizeAlgorithmConfig.CustomizedConfig = make(map[string]string)
	}
	hotPSCPUThreshold := optimizer.config.GetFloat64WithValue(config.OptimizerHotPSCPUThreshold, defaultHotPSCPUThreshold)
	conf.OptimizeAlgorithmConfig.CustomizedConfig[config.OptimizerHotPSCPUThreshold] = fmt.Sprintf("%f", hotPSCPUThreshold)

	hotPSMemoryThreshold := optimizer.config.GetFloat64WithValue(config.OptimizerHotPSMemoryThreshold, defaultHotPSMemoryThreshold)
	conf.OptimizeAlgorithmConfig.CustomizedConfig[config.OptimizerHotPSMemoryThreshold] = fmt.Sprintf("%f", hotPSMemoryThreshold)

	targetWorkerCount := optimizer.config.GetIntWithValue(config.OptimizerHotPSCPUTargetWorkerCount, defaultHotPSCPUTargetWorkerCount)
	conf.OptimizeAlgorithmConfig.CustomizedConfig[config.OptimizerHotPSCPUTargetWorkerCount] = strconv.Itoa(targetWorkerCount)

	hotPSMemoryAdjust := optimizer.config.GetIntWithValue(config.OptimizerHotPSMemoryAdjust, defaultHotPSMemoryAdjust)
	conf.OptimizeAlgorithmConfig.CustomizedConfig[config.OptimizerHotPSMemoryAdjust] = strconv.Itoa(hotPSMemoryAdjust)

	lowCPUThreshold := optimizer.config.GetFloat64WithValue(config.OptimizerLowPSCPUThreshold, defaultLowCPUThreshold)
	conf.OptimizeAlgorithmConfig.CustomizedConfig[config.OptimizerLowPSCPUThreshold] = fmt.Sprintf("%f", lowCPUThreshold)

	memoryMarginPercent := optimizer.config.GetFloat64WithValue(config.OptimizerPSMemoryMarginPercent, optimplcomm.DefaultPSMemoryMarginPercent)
	conf.OptimizeAlgorithmConfig.CustomizedConfig[config.OptimizerPSMemoryMarginPercent] = fmt.Sprintf("%f", memoryMarginPercent)

	marginCPU := optimizer.config.GetIntWithValue(config.OptimizerPSMarginCPU, defaultPSCPUMargin)
	conf.OptimizeAlgorithmConfig.CustomizedConfig[config.OptimizerPSMarginCPU] = strconv.Itoa(marginCPU)

	psCPUOverload := optimizer.config.GetFloat64WithValue(config.OptimizerPSCPUOverload, defaultPSCPUOverload)
	conf.OptimizeAlgorithmConfig.CustomizedConfig[config.OptimizerPSCPUOverload] = fmt.Sprintf("%f", psCPUOverload)

	conf.OptimizeAlgorithmConfig.Name = optalgorithm.OptimizeAlgorithmJobHotPSResource
	resOptPlan, err := optalgorithm.Optimize(optimizer.dataStore, conf.OptimizeAlgorithmConfig, optJob, nil)

	if resOptPlan == nil {
		log.Infof("Fail to optimize Hotpot PS")
		conf.OptimizeAlgorithmConfig.Name = optalgorithm.OptimizeAlgorithmJobPSResourceUtil
		resOptPlan, err = optalgorithm.Optimize(optimizer.dataStore, conf.OptimizeAlgorithmConfig, optJob, nil)
	}

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

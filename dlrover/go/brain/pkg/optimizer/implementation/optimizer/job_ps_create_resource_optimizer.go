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
	"github.com/intelligent-machine-learning/easydl/brain/pkg/datastore/dbbase"
	"github.com/intelligent-machine-learning/easydl/brain/pkg/datastore/recorder/mysql"
	optimizerapi "github.com/intelligent-machine-learning/easydl/brain/pkg/optimizer/api"
	optconfig "github.com/intelligent-machine-learning/easydl/brain/pkg/optimizer/config"
	optimplcomm "github.com/intelligent-machine-learning/easydl/brain/pkg/optimizer/implementation/common"
	"github.com/intelligent-machine-learning/easydl/brain/pkg/optimizer/implementation/optalgorithm"
	"github.com/intelligent-machine-learning/easydl/brain/pkg/utils"
	"strconv"
	"time"
)

const (
	// JobPSCreateResourceOptimizerName is the name of optimizer which optimize created job ps resources
	JobPSCreateResourceOptimizerName = "job_ps_create_resource_optimizer"
	defaultColdPSCPU                 = 12
	defaultColdPSReplica             = 4
	defaultColdPSMemory              = 16 * 1024 * 1024 * 1024 // 16GB
)

// JobPSCreateResourceOptimizer is the optimizer which optimizes creating job ps
type JobPSCreateResourceOptimizer struct {
	dataStore datastoreapi.DataStore
	config    *config.Config
}

func init() {
	registerNewOptimizerFunc(JobPSCreateResourceOptimizerName, newJobPSCreateResourceOptimizer)
}

func newJobPSCreateResourceOptimizer(dataStore datastoreapi.DataStore, config *config.Config) optimizerapi.Optimizer {
	return &JobPSCreateResourceOptimizer{
		dataStore: dataStore,
		config:    config,
	}
}

// Optimize optimizes the ps initial resources
func (optimizer *JobPSCreateResourceOptimizer) Optimize(conf *optconfig.OptimizerConfig, jobMetas []*common.JobMeta) ([]*common.OptimizePlan, error) {
	if conf == nil || conf.OptimizeAlgorithmConfig == nil {
		err := fmt.Errorf("invalid optimizer config: %v", conf)
		return nil, err
	}
	if len(jobMetas) != 1 {
		return nil, nil
	}

	jobMeta := jobMetas[0]
	// get the user of the job
	cond := &datastoreapi.Condition{
		Type: common.TypeGetDataGetJob,
		Extra: &mysql.JobCondition{
			UID: jobMeta.UUID,
		},
	}
	job := &mysql.Job{}
	if err := optimizer.dataStore.GetData(cond, job); err != nil {
		log.Errorf("Fail to get job for %s: %v", jobMeta.UUID, err)
		return nil, err
	}

	cond = &datastoreapi.Condition{
		Type: common.TypeGetDataGetJobMetrics,
		Extra: &mysql.JobMetricsCondition{
			UID: jobMeta.UUID,
		},
	}
	jobMetrics := &mysql.JobMetrics{}
	if err := optimizer.dataStore.GetData(cond, jobMetrics); err != nil {
		log.Errorf("Fail to get job metrics for %s: %v", jobMeta.UUID, err)
		return nil, err
	}

	if conf.OptimizeAlgorithmConfig.CustomizedConfig == nil {
		conf.OptimizeAlgorithmConfig.CustomizedConfig = make(map[string]string)
	}
	minPSCPU := optimizer.config.GetIntWithValue(config.OptimizerPSMinCPUCore, optimplcomm.DefaultPSMinCPU)
	conf.OptimizeAlgorithmConfig.CustomizedConfig[config.OptimizerPSMinCPUCore] = strconv.Itoa(minPSCPU)

	memoryMarginPercent := optimizer.config.GetFloat64WithValue(config.OptimizerPSMemoryMarginPercent, optimplcomm.DefaultPSMemoryMarginPercent)
	conf.OptimizeAlgorithmConfig.CustomizedConfig[config.OptimizerPSMemoryMarginPercent] = fmt.Sprintf("%f", memoryMarginPercent)

	cpuMarginPercent := optimizer.config.GetFloat64WithValue(config.OptimizerPSCPUMarginPercent, optimplcomm.DefaultPSCPUMarginPercent)
	conf.OptimizeAlgorithmConfig.CustomizedConfig[config.OptimizerPSCPUMarginPercent] = fmt.Sprintf("%f", cpuMarginPercent)

	listJobCond := &mysql.JobCondition{
		Scenario: job.Scenario,
	}
	if !job.CreatedAt.IsZero() {
		queryBackwardTimePeriodInHour := optimizer.config.GetInt(config.QueryBackwardTimePeriodInHour)
		if queryBackwardTimePeriodInHour == 0 {
			queryBackwardTimePeriodInHour = optimplcomm.DefaultQueryBackwardTimePeriodInHour
		}

		startTime := job.CreatedAt.Add(time.Duration(-1*queryBackwardTimePeriodInHour) * time.Hour)
		listJobCond.CreatedAtRange = &dbbase.TimeRange{
			From: startTime,
			To:   job.CreatedAt,
		}
	}

	cond = &datastoreapi.Condition{
		Type: common.TypeGetDataListJob,
	}
	cond.Extra = listJobCond
	historyJobs := make([]*mysql.Job, 0)
	if err := optimizer.dataStore.GetData(cond, &historyJobs); err != nil {
		log.Errorf("Fail to list job for %s with %s: %v", job.Name, job.Scenario, err)
		return nil, err
	}

	optJob := &common.OptimizeJobMeta{
		JobMeta: jobMeta,
		Metrics: utils.ConvertDBJobMetricsToJobMetrics(jobMetrics),
	}

	historyOptJobMetas := make([]*common.OptimizeJobMeta, 0)
	for _, j := range historyJobs {
		if j.UID == jobMeta.UUID {
			continue
		}
		cond = &datastoreapi.Condition{
			Type: common.TypeGetDataGetJobMetrics,
			Extra: &mysql.JobMetricsCondition{
				UID: j.UID,
			},
		}
		metrics := &mysql.JobMetrics{}
		if err := optimizer.dataStore.GetData(cond, metrics); err != nil {
			log.Errorf("fail to get job metrics for %s: %v", j.Name, err)
			continue
		}

		historyOptJobMeta := &common.OptimizeJobMeta{
			Metrics: utils.ConvertDBJobMetricsToJobMetrics(metrics),
		}
		historyOptJobMetas = append(historyOptJobMetas, historyOptJobMeta)
	}

	conf.OptimizeAlgorithmConfig.Name = optalgorithm.OptimizeAlgorithmJobPSCreateResource
	resOptPlan, err := optalgorithm.Optimize(optimizer.dataStore, conf.OptimizeAlgorithmConfig, optJob, historyOptJobMetas)
	if err != nil {
		return nil, err
	}

	if resOptPlan == nil {
		log.Infof("Fail to get resource From %s", optalgorithm.OptimizeAlgorithmJobPSCreateResource)
		coldReplica := optimizer.config.GetIntWithValue(config.OptimizerPSColdReplica, defaultColdPSReplica)
		conf.OptimizeAlgorithmConfig.CustomizedConfig[config.OptimizerPSColdReplica] = strconv.Itoa(coldReplica)

		coldCPU := optimizer.config.GetIntWithValue(config.OptimizerPSColdCPU, defaultColdPSCPU)
		conf.OptimizeAlgorithmConfig.CustomizedConfig[config.OptimizerPSColdCPU] = fmt.Sprintf("%f", float64(coldCPU))

		coldMemory := optimizer.config.GetIntWithValue(config.OptimizerPSColdMemory, defaultColdPSMemory)
		conf.OptimizeAlgorithmConfig.CustomizedConfig[config.OptimizerPSColdMemory] = fmt.Sprintf("%f", float64(coldMemory))

		cpuMargin := optimizer.config.GetFloat64WithValue(config.JobNodeCPUMargin, optimplcomm.DefaultCPUMargin)
		conf.OptimizeAlgorithmConfig.CustomizedConfig[config.JobNodeCPUMargin] = fmt.Sprintf("%f", cpuMargin)

		maxCount := optimizer.config.GetFloat64WithValue(config.OptimizerPSMaxCount, optimplcomm.DefaultMaxPSCount)
		conf.OptimizeAlgorithmConfig.CustomizedConfig[config.OptimizerPSMaxCount] = fmt.Sprintf("%f", maxCount)

		conf.OptimizeAlgorithmConfig.Name = optalgorithm.OptimizeAlgorithmJobPSColdCreateResource

		resOptPlan, err = optalgorithm.Optimize(optimizer.dataStore, conf.OptimizeAlgorithmConfig, optJob, historyOptJobMetas)
		if err != nil {
			return nil, err
		} else if resOptPlan == nil {
			return nil, nil
		}
	}

	if resOptPlan == nil || resOptPlan.JobRes == nil {
		return nil, nil
	}

	plan := &common.OptimizePlan{
		JobMeta:    jobMeta,
		AlgOptPlan: resOptPlan,
	}
	return []*common.OptimizePlan{plan}, nil
}

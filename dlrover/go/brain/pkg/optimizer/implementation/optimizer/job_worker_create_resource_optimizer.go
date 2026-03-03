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
	"github.com/intelligent-machine-learning/easydl/brain/pkg/datastore/dbbase"
	"github.com/intelligent-machine-learning/easydl/brain/pkg/datastore/recorder/mysql"
	optimizerapi "github.com/intelligent-machine-learning/easydl/brain/pkg/optimizer/api"
	optconfig "github.com/intelligent-machine-learning/easydl/brain/pkg/optimizer/config"
	optimplcomm "github.com/intelligent-machine-learning/easydl/brain/pkg/optimizer/implementation/common"
	"github.com/intelligent-machine-learning/easydl/brain/pkg/optimizer/implementation/optalgorithm"
	"github.com/intelligent-machine-learning/easydl/brain/pkg/utils"
	"time"
)

const (
	// JobWorkerCreateResourceOptimizerName is the name of optimizer to optimize the resources of the first worker
	JobWorkerCreateResourceOptimizerName = "job_worker_create_resource_optimizer"

	defaultOptimizerWorkerOomMemoryMarginPercent = 0.2
	defaultOptimizerWorkerOomMemoryMinIncrease   = 4000000000
)

// JobWorkerCreateResourceOptimizer is the optimizer which optimizes the resources of job's first worker
type JobWorkerCreateResourceOptimizer struct {
	dataStore datastoreapi.DataStore
	conf      *config.Config
}

func init() {
	registerNewOptimizerFunc(JobWorkerCreateResourceOptimizerName, newJobWorkerCreateResourceOptimizer)
}

func newJobWorkerCreateResourceOptimizer(dataStore datastoreapi.DataStore, config *config.Config) optimizerapi.Optimizer {
	return &JobWorkerCreateResourceOptimizer{
		dataStore: dataStore,
		conf:      config,
	}
}

// Optimize optimizes the chief worker initial resources
func (optimizer *JobWorkerCreateResourceOptimizer) Optimize(conf *optconfig.OptimizerConfig, jobMetas []*common.JobMeta) ([]*common.OptimizePlan, error) {
	if len(jobMetas) == 0 {
		return nil, nil
	}

	jobMeta := jobMetas[0]
	cond := &datastoreapi.Condition{
		Type: common.TypeGetDataGetJob,
		Extra: &mysql.JobCondition{
			UID: jobMeta.UUID,
		},
	}
	job := &mysql.Job{}
	err := optimizer.dataStore.GetData(cond, job)
	if err != nil {
		log.Errorf("Fail to get job from db for %s: %v", jobMeta.Name, err)
		return nil, err
	}

	cond = &datastoreapi.Condition{
		Type: common.TypeGetDataGetJobMetrics,
		Extra: &mysql.JobMetricsCondition{
			UID: jobMeta.UUID,
		},
	}
	jobMetrics := &mysql.JobMetrics{}
	err = optimizer.dataStore.GetData(cond, jobMetrics)
	if err != nil {
		log.Errorf("fail to get job metrics from db for %s: %v", jobMeta.Name, err)
		return nil, err
	}

	// Get completed jobs which share the same extra identifier
	queryBackwardTimePeriodInHour := optimizer.conf.GetIntWithValue(config.QueryBackwardTimePeriodInHour, optimplcomm.DefaultQueryBackwardTimePeriodInHour)
	now := time.Now()
	startTime := now.Add(time.Duration(-1*queryBackwardTimePeriodInHour) * time.Hour)

	cond = &datastoreapi.Condition{
		Type: common.TypeGetDataListJob,
		Extra: &mysql.JobCondition{
			Scenario: job.Scenario,
			CreatedAtRange: &dbbase.TimeRange{
				From: startTime,
				To:   now,
			},
		},
	}
	historyJobs := make([]*mysql.Job, 0)
	err = optimizer.dataStore.GetData(cond, &historyJobs)
	if err != nil {
		log.Errorf("Fail to list job for %s: %v", job.Scenario, err)
		return nil, err
	}

	dbHistoryJobsMetrics := make([]*mysql.JobMetrics, 0)
	for _, historyJob := range historyJobs {
		historyJobMetrics := &mysql.JobMetrics{}
		cond = &datastoreapi.Condition{
			Type: common.TypeGetDataGetJobMetrics,
			Extra: &mysql.JobMetricsCondition{
				UID: historyJob.UID,
			},
		}
		err = optimizer.dataStore.GetData(cond, historyJobMetrics)
		if err != nil {
			log.Errorf("fail to get job metrics for %s: %v", historyJob.Name, err)
			continue
		}
		dbHistoryJobsMetrics = append(dbHistoryJobsMetrics, historyJobMetrics)
	}

	if conf.OptimizeAlgorithmConfig == nil {
		conf.OptimizeAlgorithmConfig = &optconfig.OptimizeAlgorithmConfig{}
	}
	if conf.OptimizeAlgorithmConfig.CustomizedConfig == nil {
		conf.OptimizeAlgorithmConfig.CustomizedConfig = make(map[string]string)
	}

	historyJobsMetrics := make([]*common.OptimizeJobMeta, 0)
	for _, j := range dbHistoryJobsMetrics {
		historyJob := &common.OptimizeJobMeta{
			Metrics: utils.ConvertDBJobMetricsToJobMetrics(j),
		}
		historyJobsMetrics = append(historyJobsMetrics, historyJob)
	}

	optJob := &common.OptimizeJobMeta{
		JobMeta: jobMeta,
		Metrics: utils.ConvertDBJobMetricsToJobMetrics(jobMetrics),
	}

	oom := false
	jobStatus := &common.JobStatus{}
	err = json.Unmarshal([]byte(job.Status), jobStatus)
	if err != nil {
		log.Warningf("fail to unmarshal status for %s with %s: %v", job.Name, job.Status, err)
	} else {
		oom = jobStatus.IsOOM
	}

	if oom {
		conf.OptimizeAlgorithmConfig.Name = optalgorithm.OptimizeAlgorithmJobWorkerCreateOomResource

		oomMemoryMarginPercent := optimizer.conf.GetFloat64WithValue(config.OptimizerWorkerOomMemoryMarginPercent, defaultOptimizerWorkerOomMemoryMarginPercent)
		conf.OptimizeAlgorithmConfig.CustomizedConfig[config.OptimizerWorkerOomMemoryMarginPercent] = fmt.Sprintf("%f", oomMemoryMarginPercent)
		oomMemoryMinIncrease := optimizer.conf.GetFloat64WithValue(config.OptimizerWorkerOomMemoryMinIncrease, defaultOptimizerWorkerOomMemoryMinIncrease)
		conf.OptimizeAlgorithmConfig.CustomizedConfig[config.OptimizerWorkerOomMemoryMinIncrease] = fmt.Sprintf("%f", oomMemoryMinIncrease)
	} else {
		conf.OptimizeAlgorithmConfig.Name = optalgorithm.OptimizeAlgorithmJobWorkerCreateResource

		memoryMarginPercent := optimizer.conf.GetFloat64WithValue(config.JobNodeMemoryMarginPercent, optimplcomm.DefaultMemoryMarginPercent)
		conf.OptimizeAlgorithmConfig.CustomizedConfig[config.JobNodeMemoryMarginPercent] = fmt.Sprintf("%f", memoryMarginPercent)
		minChiefCPUCore := optimizer.conf.GetFloat64WithValue(config.OptimizerMinWorkerCreateCPU, optimplcomm.DefaultWorkerCreateCPU)
		conf.OptimizeAlgorithmConfig.CustomizedConfig[config.OptimizerMinWorkerCreateCPU] = fmt.Sprintf("%f", minChiefCPUCore)
	}

	resOptPlan, err := optalgorithm.Optimize(optimizer.dataStore, conf.OptimizeAlgorithmConfig, optJob, historyJobsMetrics)

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

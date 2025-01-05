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

package mysql

import (
	"errors"
	"fmt"
)

// JobMetricsFakeRecorder is the fake recorder struct of job metrics
type JobMetricsFakeRecorder struct {
	records map[string]*JobMetrics
}

// NewJobMetricsFakeRecorder returns a new fake job metrics recorder
func NewJobMetricsFakeRecorder() JobMetricsRecorderInterface {
	return &JobMetricsFakeRecorder{
		records: make(map[string]*JobMetrics),
	}
}

func canApplyJobMetricsCondition(condition *JobMetricsCondition, jobMetrics *JobMetrics) bool {
	if len(condition.UID) > 0 && condition.UID != jobMetrics.UID {
		return false
	}
	return true
}

// Get return a row
func (r *JobMetricsFakeRecorder) Get(condition *JobMetricsCondition, job *JobMetrics) error {
	if job == nil {
		job = &JobMetrics{}
	}
	for _, jobMetrics := range r.records {
		if canApplyJobMetricsCondition(condition, jobMetrics) {
			*job = *jobMetrics
			return nil
		}
	}
	return fmt.Errorf("fail to find record for %v", condition)
}

// List return multiple row
func (r *JobMetricsFakeRecorder) List(condition *JobMetricsCondition, jobs *[]*JobMetrics) error {
	if jobs == nil {
		records := make([]*JobMetrics, 0)
		jobs = &records
	}
	for _, jobMetrics := range r.records {
		if canApplyJobMetricsCondition(condition, jobMetrics) {
			*jobs = append(*jobs, jobMetrics)
		}
	}
	return nil
}

// Upsert insert or update a row
func (r *JobMetricsFakeRecorder) Upsert(jobMetrics *JobMetrics) error {
	if len(jobMetrics.UID) == 0 {
		return errors.New("JobUUID can not be empty")
	}
	r.records[jobMetrics.UID] = jobMetrics
	return nil
}

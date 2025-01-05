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
	"github.com/intelligent-machine-learning/easydl/brain/pkg/datastore/dbbase"
	"xorm.io/xorm"
)

// TableJobMetrics is the name of job metrics table
const TableJobMetrics = "job_metrics"

// JobMetricsCondition is the struct of sql condition for job metrics table
type JobMetricsCondition struct {
	UID string
}

// JobMetrics is the struct of job metrics for mysql db
type JobMetrics struct {
	UID                string
	HyperParamsFeature string
	JobFeature         string
	DatasetFeature     string
	ModelFeature       string
	JobRuntime         string
	ExitReason         string
	Optimization       string
	Resource           string
	CustomizedData     string
	Type               string
}

// Apply applies JobMetricsCondition
func (c *JobMetricsCondition) Apply(session *xorm.Session) *xorm.Session {
	if c.UID != "" {
		session.Where("uid = ?", c.UID)
	}
	return session
}

// JobMetricsRecorderInterface is the recorder interface of job metrics
type JobMetricsRecorderInterface interface {
	Get(condition *JobMetricsCondition, job *JobMetrics) error
	List(condition *JobMetricsCondition, jobs *[]*JobMetrics) error
	Upsert(jobMetrics *JobMetrics) error
}

// JobMetricsRecorder is the recorder struct of job metrics
type JobMetricsRecorder struct {
	Recorder dbbase.RecorderInterface
}

// NewJobMetricsDBRecorder creates a new JobMetricsRecorder
func NewJobMetricsDBRecorder(db *dbbase.Database) JobMetricsRecorderInterface {
	return &JobMetricsRecorder{
		Recorder: &dbbase.DatabaseRecorder{Engine: db.Engine, TableName: TableJobMetrics},
	}
}

// Get returns a row
func (r *JobMetricsRecorder) Get(condition *JobMetricsCondition, job *JobMetrics) error {
	if job == nil {
		job = &JobMetrics{}
	}
	return r.Recorder.Get(job, condition)
}

// List returns multiple rows
func (r *JobMetricsRecorder) List(condition *JobMetricsCondition, jobs *[]*JobMetrics) error {
	if jobs == nil {
		records := make([]*JobMetrics, 0)
		jobs = &records
	}
	return r.Recorder.List(jobs, condition)
}

// Upsert updates or insert a row
func (r *JobMetricsRecorder) Upsert(jobMetrics *JobMetrics) error {
	return r.Recorder.Upsert(jobMetrics)
}

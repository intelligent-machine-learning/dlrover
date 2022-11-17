// Copyright 2022 The EasyDL Authors. All rights reserved.
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
	"time"
	"xorm.io/xorm"
)

// TableJobMetrics is the name of job metrics table
const TableJobMetrics = "job_metrics"

// JobMetricsCondition is the struct of sql condition for job metrics table
type JobMetricsCondition struct {
	JobUUID        string
	JobName        string
	Namespace      string
	Cluster        string
	User           string
	CreatedAtRange *dbbase.TimeRange
	IsProd         *bool
}

// JobMetrics is the struct of job metrics
type JobMetrics struct {
	ID        int64
	JobUUID   string
	JobName   string
	Namespace string
	Cluster   string
	User      string
	CreatedAt time.Time
	// TrainingHyperParams
	MetricUserConfig string
	// WorkflowFeature
	MetricAistudioJobFeature     string
	MetricTrainingDatasetFeature string
	MetricModelFeature           string
	MetricJobRuntime             string
	IsProd                       int32
	ExitReason                   string
	Optimization                 string
	Type                         string
	Resource                     string
	CustomizedData               string
}

// Apply applies JobMetricsCondition
func (c *JobMetricsCondition) Apply(session *xorm.Session) *xorm.Session {
	if c.JobUUID != "" {
		session.Where("job_uuid = ?", c.JobUUID)
	}
	if c.JobName != "" {
		session.Where("job_name = ?", c.JobName)
	}
	if c.Namespace != "" {
		session.Where("namespace = ?", c.Namespace)
	}
	if c.Cluster != "" {
		session.Where("cluster = ?", c.Cluster)
	}
	if c.User != "" {
		session.Where("user = ?", c.User)
	}
	if r := c.CreatedAtRange; r != nil {
		if !r.From.IsZero() {
			session.Where("created_at >= ?", r.From)
		}
		if !r.To.IsZero() {
			session.Where("created_at <= ?", r.To)
		}
	}
	if c.IsProd != nil {
		if *c.IsProd {
			session.Where("is_prod = 1")
		} else {
			session.Where("is_prod = 0")
		}
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

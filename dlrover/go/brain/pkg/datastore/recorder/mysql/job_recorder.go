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
	"time"
	"xorm.io/xorm"
)

// TableJob is the name of job table
const TableJob = "job"

// JobCondition is the struct of sql condition for job table
type JobCondition struct {
	UID            string
	Name           string
	Scenario       string
	CreatedAtRange *dbbase.TimeRange
}

// Job is the struct of job for mysql db
type Job struct {
	UID        string
	Name       string
	Scenario   string
	CreatedAt  time.Time
	StartedAt  time.Time
	FinishedAt time.Time
	Status     string
}

// Apply applies JobCondition
func (c *JobCondition) Apply(session *xorm.Session) *xorm.Session {
	if c.UID != "" {
		session.Where("uid = ?", c.UID)
	}
	if c.Name != "" {
		session.Where("name = ?", c.Name)
	}
	if c.Scenario != "" {
		session.Where("scenario = ?", c.Scenario)
	}
	if r := c.CreatedAtRange; r != nil {
		if !r.From.IsZero() {
			session.Where("created_at >= ?", r.From)
		}
		if !r.To.IsZero() {
			session.Where("created_at <= ?", r.To)
		}
	}
	return session
}

// JobRecorderInterface is the recorder interface of job
type JobRecorderInterface interface {
	Get(condition *JobCondition, job *Job) error
	List(condition *JobCondition, jobs *[]*Job) error
	Upsert(job *Job) error
}

// JobRecorder is the recorder struct of job
type JobRecorder struct {
	Recorder dbbase.RecorderInterface
}

// NewJobDBRecorder creates a new JobRecorder
func NewJobDBRecorder(db *dbbase.Database) JobRecorderInterface {
	return &JobRecorder{
		Recorder: &dbbase.DatabaseRecorder{Engine: db.Engine, TableName: TableJob},
	}
}

// Get returns a row
func (r *JobRecorder) Get(condition *JobCondition, job *Job) error {
	if job == nil {
		job = &Job{}
	}
	return r.Recorder.Get(job, condition)
}

// List returns multiple rows
func (r *JobRecorder) List(condition *JobCondition, jobs *[]*Job) error {
	if jobs == nil {
		records := make([]*Job, 0)
		jobs = &records
	}
	return r.Recorder.List(jobs, condition)
}

// Upsert updates or insert a row
func (r *JobRecorder) Upsert(job *Job) error {
	return r.Recorder.Upsert(job)
}

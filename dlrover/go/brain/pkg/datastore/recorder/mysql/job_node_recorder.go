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

// TableJobNode is the name of job node table
const TableJobNode = "job_node"

// JobNodeCondition is the struct of sql condition for job node table
type JobNodeCondition struct {
	UID            string
	Name           string
	JobUUID        string
	JobName        string
	Type           string
	CreatedAtRange *dbbase.TimeRange
}

// JobNode is the struct of job node for mysql db
type JobNode struct {
	UID            string
	Name           string
	JobUUID        string
	JobName        string
	Type           string
	CreatedAt      time.Time
	StartedAt      time.Time
	FinishedAt     time.Time
	Resource       string
	Status         string
	CustomizedData string
}

// Apply applies JobNodeCondition
func (c *JobNodeCondition) Apply(session *xorm.Session) *xorm.Session {
	if c.UID != "" {
		session.Where("uid = ?", c.UID)
	}
	if c.Name != "" {
		session.Where("name = ?", c.Name)
	}
	if c.JobUUID != "" {
		session.Where("job_uuid = ?", c.JobUUID)
	}
	if c.JobName != "" {
		session.Where("job_name = ?", c.JobName)
	}
	if c.Type != "" {
		session.Where("type = ?", c.Type)
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

// JobNodeRecorderInterface is the recorder interface of job node
type JobNodeRecorderInterface interface {
	Get(condition *JobNodeCondition, job *JobNode) error
	List(condition *JobNodeCondition, jobs *[]*JobNode) error
	Upsert(job *JobNode) error
}

// JobNodeRecorder is the recorder struct of job node
type JobNodeRecorder struct {
	Recorder dbbase.RecorderInterface
}

// NewJobNodeDBRecorder creates a new JobNodeRecorder
func NewJobNodeDBRecorder(db *dbbase.Database) JobNodeRecorderInterface {
	return &JobNodeRecorder{
		Recorder: &dbbase.DatabaseRecorder{Engine: db.Engine, TableName: TableJobNode},
	}
}

// Get returns a row
func (r *JobNodeRecorder) Get(condition *JobNodeCondition, node *JobNode) error {
	if node == nil {
		node = &JobNode{}
	}
	return r.Recorder.Get(node, condition)
}

// List returns multiple rows
func (r *JobNodeRecorder) List(condition *JobNodeCondition, nodes *[]*JobNode) error {
	if nodes == nil {
		records := make([]*JobNode, 0)
		nodes = &records
	}
	return r.Recorder.List(nodes, condition)
}

// Upsert updates or insert a row
func (r *JobNodeRecorder) Upsert(node *JobNode) error {
	return r.Recorder.Upsert(node)
}

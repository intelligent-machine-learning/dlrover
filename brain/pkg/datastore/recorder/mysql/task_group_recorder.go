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

// tableTaskGroups is the name of job metrics table
const tableTaskGroups = "task_groups"

// TaskGroup represents a task structure
type TaskGroup struct {
	ID               int64
	JobUUID          string
	JobName          string
	JobType          string
	Namespace        string
	Cluster          string
	Name             string
	ProfileID        string
	User             string
	TaskNumRequested int32
	CPURequested     float32
	MemoryRequested  int64
	DiskRequested    int64
	GPUTypeRequested string
	GPURequested     int32
	HBOTaskNum       int32   `xorm:"hbo_task_num"`
	HBOCPU           float32 `xorm:"hbo_cpu"`
	HBOMemory        int64   `xorm:"hbo_memory"`
	HBODisk          int64   `xorm:"hbo_disk"`
	HBOGPUType       string  `xorm:"hbo_gpu_type"`
	HBOGPU           int32   `xorm:"hbo_gpu"`
	CurrentTaskNum   int32
	CurrentCPU       float32
	CurrentMemory    int64
	CurrentDisk      int64
	CurrentGPUType   string
	CurrentGPU       int32
	CreatedAt        time.Time
	StartedAt        time.Time
	FinishedAt       time.Time
	RemovedAt        time.Time
	UpdatedAt        time.Time
}

// TaskGroupCondition is the struct task group condition
type TaskGroupCondition struct {
	JobUUID        string
	JobName        string
	Namespace      string
	Cluster        string
	Name           string
	User           string
	CreatedAtRange *dbbase.TimeRange
}

// Apply applies TaskGroupCondition
func (c *TaskGroupCondition) Apply(session *xorm.Session) *xorm.Session {
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
	if c.Name != "" {
		session.Where("name = ?", c.Name)
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
	return session
}

// TaskGroupsRecorderInterface is the recorder interface of job metrics
type TaskGroupsRecorderInterface interface {
	Get(condition *TaskGroupCondition, group *TaskGroup) error
	List(condition *TaskGroupCondition, groups *[]*TaskGroup) error
	Upsert(taskGroup *TaskGroup) error
}

// TaskGroupsRecorder is the recorder struct of job metrics
type TaskGroupsRecorder struct {
	Recorder dbbase.RecorderInterface
}

// NewTaskGroupsDBRecorder creates a new TaskGroupsRecorder
func NewTaskGroupsDBRecorder(db *dbbase.DB) TaskGroupsRecorderInterface {
	return &TaskGroupsRecorder{
		Recorder: &dbbase.DBRecorder{Engine: db.Engine, TableName: tableTaskGroups},
	}
}

// Get returns a row
func (r *TaskGroupsRecorder) Get(condition *TaskGroupCondition, group *TaskGroup) error {
	if group == nil {
		group = &TaskGroup{}
	}
	return r.Recorder.Get(group, condition)
}

// List returns multiple rows
func (r *TaskGroupsRecorder) List(condition *TaskGroupCondition, groups *[]*TaskGroup) error {
	if groups == nil {
		records := make([]*TaskGroup, 0)
		groups = &records
	}
	return r.Recorder.List(groups, condition)
}

// Upsert insert or update a row
func (r *TaskGroupsRecorder) Upsert(taskGroup *TaskGroup) error {
	return r.Recorder.Upsert(taskGroup)
}

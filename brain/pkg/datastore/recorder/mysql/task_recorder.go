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

const tableTasks = "tasks"

// Task represents a task structure
type Task struct {
	ID                  int64
	UUID                string
	JobUUID             string
	Name                string
	Namespace           string
	MainContainer       string
	Cluster             string
	Status              string
	User                string
	Component           string
	Content             string
	OOM                 bool
	TaskGroupName       string
	IP                  string
	CreatedAt           time.Time
	StartedAt           time.Time
	FinishedAt          time.Time
	RemovedAt           time.Time
	UpdatedAt           time.Time
	CPU                 float32
	Memory              int64
	Disk                int64
	GPUType             string
	GPU                 int32
	MainContainerCPU    float32
	MainContainerMemory int64
	MaxCPUUtility       float32
	MinCPUUtility       float32
	SumCPUUtility       float32
	SumCPUUsedCores     float32
	MaxMemoryUtility    float32
	MinMemoryUtility    float32
	SumMemoryUsedCores  float32
	SumMemoryUtility    float32
	RecordCount         int32
	UtilityEndTime      time.Time
	QoS                 string `xorm:"qos"`
	VPAOpStatus         string `xorm:"vpa_op_status"`
	InstanceInfo        string
	NodeName            string
	NodeIP              string
	Hostname            string
}

// TaskCondition is the struct task condition
type TaskCondition struct {
	UUID            string
	Name            string
	Namespace       string
	Cluster         string
	User            string
	Status          string
	OOM             *bool
	JobUUID         string
	TaskGroupName   string
	QoS             string
	IP              string
	NodeName        string
	NodeIP          string
	CreatedAtRange  *dbbase.TimeRange
	StartedAtRange  *dbbase.TimeRange
	FinishedAtRange *dbbase.TimeRange
	RemovedAtRange  *dbbase.TimeRange
}

// Apply applies task condition on query
func (c *TaskCondition) Apply(session *xorm.Session) *xorm.Session {
	if c.UUID != "" {
		session.Where("uuid = ?", c.UUID)
	}
	if c.JobUUID != "" {
		session.Where("job_uuid = ?", c.JobUUID)
	}
	if c.Name != "" {
		session.Where("name = ?", c.Name)
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
	if r := c.StartedAtRange; r != nil {
		if !r.From.IsZero() {
			session.Where("started_at >= ?", r.From)
		}
		if !r.To.IsZero() {
			session.Where("started_at <= ?", r.To)
		}
	}
	if r := c.FinishedAtRange; r != nil {
		if !r.From.IsZero() {
			session.Where("finished_at >= ?", r.From)
		}
		if !r.To.IsZero() {
			session.Where("finished_at <= ?", r.To)
		}
	}
	if r := c.RemovedAtRange; r != nil {
		if !r.From.IsZero() {
			session.Where("removed_at >= ?", r.From)
		}
		if !r.To.IsZero() {
			session.Where("removed_at <= ?", r.To)
		}
	}
	if c.Status != "" {
		session.Where("status = ?", c.Status)
	}
	if c.OOM != nil {
		session.Where("oom = ?", *c.OOM)
	}
	if c.TaskGroupName != "" {
		session.Where("task_group_name = ?", c.TaskGroupName)
	}
	if c.QoS != "" {
		session.Where("qos = ?", c.QoS)
	}
	if c.IP != "" {
		session.Where("ip = ?", c.IP)
	}
	if c.NodeIP != "" {
		session.Where("node_ip = ?", c.NodeIP)
	}
	if c.NodeName != "" {
		session.Where("node_name = ?", c.NodeName)
	}
	return session
}

// TasksRecorderInterface is the recorder interface of tasks
type TasksRecorderInterface interface {
	List(condition *TaskCondition, tasks *[]*Task) error
	Get(condition *TaskCondition, task *Task) error
	Upsert(task *Task) error
}

// TasksRecorder is the recorder struct of tasks
type TasksRecorder struct {
	Recorder dbbase.RecorderInterface
}

// NewTasksRecorder returns a new TasksRecorder
func NewTasksRecorder(db *dbbase.DB) TasksRecorderInterface {
	return &TasksRecorder{
		Recorder: &dbbase.DBRecorder{Engine: db.Engine, TableName: tableTasks},
	}
}

// List lists tasks using the given parameters
func (r *TasksRecorder) List(condition *TaskCondition, tasks *[]*Task) error {
	if tasks == nil {
		records := make([]*Task, 0)
		tasks = &records
	}
	return r.Recorder.List(tasks, condition)
}

// Get one task from DB. If multiple records match, return the one with the highest id
func (r *TasksRecorder) Get(condition *TaskCondition, task *Task) error {
	if task == nil {
		task = &Task{}
	}
	return r.Recorder.Get(task, condition)
}

// Upsert insert or update a row
func (r *TasksRecorder) Upsert(task *Task) error {
	return r.Recorder.Upsert(task)
}

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

// JobFakeRecorder is the fake recorder struct of job
type JobFakeRecorder struct {
	records map[string]*Job
}

// NewJobFakeRecorder returns a new fake job recorder
func NewJobFakeRecorder() JobRecorderInterface {
	return &JobFakeRecorder{
		records: make(map[string]*Job),
	}
}

func canApplyJobCondition(c *JobCondition, job *Job) bool {
	if len(c.UID) > 0 && c.UID != job.UID {
		return false
	}
	if len(c.Name) > 0 && c.Name != job.Name {
		return false
	}
	if len(c.Scenario) > 0 && c.Scenario != job.Scenario {
		return false
	}

	if c.CreatedAtRange != nil {
		if c.CreatedAtRange.From.After(job.CreatedAt) {
			return false
		}
		if c.CreatedAtRange.To.Before(job.CreatedAt) {
			return false
		}
	}
	return true
}

// Get return a row
func (r *JobFakeRecorder) Get(condition *JobCondition, job *Job) error {
	if job == nil {
		job = &Job{}
	}
	for _, record := range r.records {
		if canApplyJobCondition(condition, record) {
			*job = *record
			return nil
		}
	}
	return fmt.Errorf("fail to find record for %v", condition)
}

// List return multiple row
func (r *JobFakeRecorder) List(condition *JobCondition, jobs *[]*Job) error {
	if jobs == nil {
		records := make([]*Job, 0)
		jobs = &records
	}
	for _, job := range r.records {
		if canApplyJobCondition(condition, job) {
			*jobs = append(*jobs, job)
		}
	}
	return nil
}

// Upsert insert or update a row
func (r *JobFakeRecorder) Upsert(job *Job) error {
	if len(job.UID) == 0 {
		return errors.New("CID can not be empty")
	}
	r.records[job.UID] = job
	return nil
}

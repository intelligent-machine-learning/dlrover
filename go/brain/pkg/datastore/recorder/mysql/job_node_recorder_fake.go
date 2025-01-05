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

// JobNodeFakeRecorder is the fake recorder struct of job node
type JobNodeFakeRecorder struct {
	records map[string]*JobNode
}

// NewJobNodeFakeRecorder returns a new fake job node recorder
func NewJobNodeFakeRecorder() JobNodeRecorderInterface {
	return &JobNodeFakeRecorder{
		records: make(map[string]*JobNode),
	}
}

func canApplyJobNodeCondition(c *JobNodeCondition, node *JobNode) bool {
	if len(c.UID) > 0 && c.UID != node.UID {
		return false
	}
	if len(c.JobUUID) > 0 && c.JobUUID != node.JobUUID {
		return false
	}
	if len(c.JobName) > 0 && c.JobName != node.JobName {
		return false
	}
	if len(c.Type) > 0 && c.Type != node.Type {
		return false
	}

	if c.CreatedAtRange != nil {
		if c.CreatedAtRange.From.After(node.CreatedAt) {
			return false
		}
		if c.CreatedAtRange.To.Before(node.CreatedAt) {
			return false
		}
	}
	return true
}

// Get return a row
func (r *JobNodeFakeRecorder) Get(condition *JobNodeCondition, node *JobNode) error {
	if node == nil {
		node = &JobNode{}
	}
	for _, jobNode := range r.records {
		if canApplyJobNodeCondition(condition, node) {
			*node = *jobNode
			return nil
		}
	}
	return fmt.Errorf("fail to find record for %v", condition)
}

// List return multiple row
func (r *JobNodeFakeRecorder) List(condition *JobNodeCondition, nodes *[]*JobNode) error {
	if nodes == nil {
		records := make([]*JobNode, 0)
		nodes = &records
	}
	for _, jobNode := range r.records {
		if canApplyJobNodeCondition(condition, jobNode) {
			*nodes = append(*nodes, jobNode)
		}
	}
	return nil
}

// Upsert insert or update a row
func (r *JobNodeFakeRecorder) Upsert(node *JobNode) error {
	if len(node.UID) == 0 {
		return errors.New("CID can not be empty")
	}
	r.records[node.UID] = node
	return nil
}

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

import "github.com/intelligent-machine-learning/easydl/brain/pkg/datastore/dbbase"

// Client is the struct of mysql db client
type Client struct {
	JobMetricsRecorder JobMetricsRecorderInterface
}

// NewClient returns a new mysql db client
// Test
func NewClient(db *dbbase.DatabaseRecorder) *Client {
	return &Client{
		JobMetricsRecorder: NewJobMetricsDBRecorder(db),
	}
}

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
	log "github.com/golang/glog"
	"github.com/intelligent-machine-learning/easydl/brain/pkg/config"
	"github.com/intelligent-machine-learning/easydl/brain/pkg/datastore/dbbase"
)

// Client is the struct of mysql db client
type Client struct {
	JobMetricsRecorder JobMetricsRecorderInterface
	JobRecorder        JobRecorderInterface
	JobNodeRecorder    JobNodeRecorderInterface
}

// NewClient returns a new mysql db client
func NewClient(conf *config.Config) *Client {
	user := conf.GetString(config.DBUser)
	pw := conf.GetString(config.DBPassword)
	engineType := conf.GetString(config.DBEngineType)
	url := conf.GetString(config.DBURL)

	log.Infof("create mysql db with user(%s), password(%s), engineType(%s), url(%s)", user, pw, engineType, url)

	db := dbbase.NewDatabase(user, pw, engineType, url)
	return &Client{
		JobMetricsRecorder: NewJobMetricsDBRecorder(db),
		JobRecorder:        NewJobDBRecorder(db),
		JobNodeRecorder:    NewJobNodeDBRecorder(db),
	}
}

// NewFakeClient returns a new fake mysql db client
func NewFakeClient() *Client {
	return &Client{
		JobMetricsRecorder: NewJobMetricsFakeRecorder(),
		JobRecorder:        NewJobFakeRecorder(),
		JobNodeRecorder:    NewJobNodeFakeRecorder(),
	}
}

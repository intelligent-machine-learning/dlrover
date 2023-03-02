// Copyright 2023 The DLRover Authors. All rights reserved.
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

package k8s

import (
	"fmt"
	"github.com/intelligent-machine-learning/easydl/brain/pkg/config"
	dsimpl "github.com/intelligent-machine-learning/easydl/brain/pkg/datastore/implementation"
	"github.com/intelligent-machine-learning/easydl/brain/pkg/datastore/recorder/mysql"
	pb "github.com/intelligent-machine-learning/easydl/brain/pkg/proto"
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestDatastore(t *testing.T) {
	if !toRunDataStoreTest {
		return
	}

	// run "k8s service mysql --url -n dlrover" to get the port number
	port := "61680"
	url := fmt.Sprintf("tcp(127.0.0.1:%s)/dlrover?parseTime=true&interpolateParams=true&loc=Local", port)

	conf := config.NewEmptyConfig()
	conf.Set(config.DBUser, "root")
	conf.Set(config.DBPassword, "root")
	conf.Set(config.DBEngineType, "mysql")
	conf.Set(config.DBURL, url)
	dbClient := mysql.NewClient(conf)

	baseDatastore := &dsimpl.BaseDataStore{
		Client: dbClient,
	}

	jobMetrics := &pb.JobMetrics{
		JobMeta: &pb.JobMeta{
			Uuid: "testing-job-uuid",
			Name: "testing-job",
		},
		MetricsType: pb.MetricsType_Job_Exit_Reason,
		Metrics: &pb.JobMetrics_JobExitReason{
			JobExitReason: "oom",
		},
	}

	err := baseDatastore.PersistData(nil, jobMetrics, nil)
	assert.NoError(t, err)
}

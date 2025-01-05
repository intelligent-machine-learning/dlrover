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

package implementation

import (
	"fmt"
	"github.com/intelligent-machine-learning/easydl/brain/pkg/common"
	"github.com/intelligent-machine-learning/easydl/brain/pkg/config"
	datastoreapi "github.com/intelligent-machine-learning/easydl/brain/pkg/datastore/api"
	dsimplutils "github.com/intelligent-machine-learning/easydl/brain/pkg/datastore/implementation/utils"
	"github.com/intelligent-machine-learning/easydl/brain/pkg/datastore/recorder/mysql"
)

const (
	// ElasticJobDataStoreName is the name of elasticjob datastore
	ElasticJobDataStoreName = "elasticjob_datastore"
)

func init() {
	registerNewFunc(ElasticJobDataStoreName, newElasticJobDataStore)
}

// ElasticJobDataStore is the struct of elasticjob data store
type ElasticJobDataStore struct {
	client *mysql.Client
}

func newElasticJobDataStore(conf *config.Config) (datastoreapi.DataStore, error) {
	client := mysql.NewClient(conf)

	return &ElasticJobDataStore{
		client: client,
	}, nil
}

// PersistData persists data into storage
func (store *ElasticJobDataStore) PersistData(condition *datastoreapi.Condition, record interface{}, extra interface{}) error {
	switch condition.Type {
	case common.TypeUpsertJob:
		job, ok := record.(*mysql.Job)
		if !ok {
			return fmt.Errorf("record must be Job for cond %v", condition)
		}
		return store.client.JobRecorder.Upsert(job)
	case common.TypeUpsertJobNode:
		node, ok := record.(*mysql.JobNode)
		if !ok {
			return fmt.Errorf("record must be JobNode for cond %v", condition)
		}
		return store.client.JobNodeRecorder.Upsert(node)
	default:
		return fmt.Errorf("invalid type: %s", condition.Type)
	}
}

// GetData returns data for a given condition
func (store *ElasticJobDataStore) GetData(condition *datastoreapi.Condition, data interface{}) error {
	return dsimplutils.GetData(store.client, condition, data)
}

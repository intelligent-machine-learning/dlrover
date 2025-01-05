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
	"github.com/intelligent-machine-learning/easydl/brain/pkg/config"
	datastoreapi "github.com/intelligent-machine-learning/easydl/brain/pkg/datastore/api"
	dsimplutils "github.com/intelligent-machine-learning/easydl/brain/pkg/datastore/implementation/utils"
	"github.com/intelligent-machine-learning/easydl/brain/pkg/datastore/recorder/mysql"
	pb "github.com/intelligent-machine-learning/easydl/brain/pkg/proto"
)

const (

	// BaseDataStoreName is the name of base data store
	BaseDataStoreName = "base_datastore"
)

func init() {
	registerNewFunc(BaseDataStoreName, newBaseDataStore)
}

// BaseDataStore is the base data store
type BaseDataStore struct {
	Client *mysql.Client
}

func newBaseDataStore(conf *config.Config) (datastoreapi.DataStore, error) {
	client := mysql.NewClient(conf)

	return &BaseDataStore{
		Client: client,
	}, nil
}

// PersistData persists data into storage
func (store *BaseDataStore) PersistData(condition *datastoreapi.Condition, record interface{}, extra interface{}) error {
	jobMetrics, ok := record.(*pb.JobMetrics)
	if !ok {
		err := fmt.Errorf("record must be type of pb JobMetrics")
		return err
	}
	return dsimplutils.PersistJobMetrics(store.Client, jobMetrics)
}

// GetData returns data for a given condition
func (store *BaseDataStore) GetData(condition *datastoreapi.Condition, data interface{}) error {
	return dsimplutils.GetData(store.Client, condition, data)
}

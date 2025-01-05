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

package utils

import (
	log "github.com/golang/glog"
	"github.com/intelligent-machine-learning/easydl/brain/pkg/common"
	datastoreapi "github.com/intelligent-machine-learning/easydl/brain/pkg/datastore/api"
	"github.com/intelligent-machine-learning/easydl/brain/pkg/datastore/recorder/mysql"
)

// GetJobNodesByGroup returns a given job's nodes of particular group
func GetJobNodesByGroup(dataStore datastoreapi.DataStore, jobMeta *common.JobMeta, groupName string) []*mysql.JobNode {
	cond := &datastoreapi.Condition{
		Type: common.TypeGetDataListJobNode,
		Extra: &mysql.JobNodeCondition{
			JobUUID: jobMeta.UUID,
			Type:    groupName,
		},
	}

	nodes := make([]*mysql.JobNode, 0)
	err := dataStore.GetData(cond, &nodes)
	if err != nil {
		log.Errorf("Fail to get job nodes by group %v: %v", cond, err)
		return nil
	}
	return nodes
}

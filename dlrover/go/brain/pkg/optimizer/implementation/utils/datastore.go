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

package utils

import (
	"encoding/json"
	log "github.com/golang/glog"
	"github.com/intelligent-machine-learning/easydl/brain/pkg/common"
	"github.com/intelligent-machine-learning/easydl/brain/pkg/datastore/recorder/mysql"
	optimplcomm "github.com/intelligent-machine-learning/easydl/brain/pkg/optimizer/implementation/common"
	"strconv"
	"strings"
)

// GetResourceFromJobNode gets resources from job nodes
func GetResourceFromJobNode(nodes []*mysql.JobNode, resType string) map[uint64]float64 {
	if resType != optimplcomm.ResourceTypeCPU && resType != optimplcomm.ResourceTypeMemory {
		log.Errorf("invalid resource type %s", resType)
		return nil
	}

	res := make(map[uint64]float64)

	for _, node := range nodes {
		p := strings.LastIndex(node.Name, "-")
		if p < 0 {
			log.Errorf("invalid node name: %s", node.Name)
			continue
		}
		n, err := strconv.Atoi(node.Name[p+1:])
		if err != nil {
			log.Errorf("invalid node name %s: %v", node.Name, err)
			continue
		}
		nodeRes := &common.PodResource{}
		err = json.Unmarshal([]byte(node.Resource), nodeRes)
		if err != nil {
			log.Errorf("fail to unmarshal resources %s for %s: %v", node.Resource, node.Name, err)
			continue
		}

		if resType == optimplcomm.ResourceTypeCPU {
			res[uint64(n)] = float64(nodeRes.CPUCore)
		} else {
			res[uint64(n)] = nodeRes.Memory
		}
	}
	return res
}

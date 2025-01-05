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
	optimplcomm "github.com/intelligent-machine-learning/easydl/brain/pkg/optimizer/implementation/common"
)

// CalculateJobNodeAvgResources get the avg resources for each job node from the runtime infos
func CalculateJobNodeAvgResources(rts []*common.JobRuntimeInfo, sampleStep int, resType string) map[uint64]float64 {
	nRecords := len(rts)
	if sampleStep > nRecords {
		sampleStep = nRecords
	}
	nodeSumRes := make(map[uint64]float64)
	nodeRecordNum := make(map[uint64]float64)
	avgRes := make(map[uint64]float64)

	for i := 0; i < sampleStep; i++ {
		var resources map[uint64]float64
		rt := rts[nRecords-i-1]
		resources = getResourceAccordingToType(rt, resType)
		if resources == nil {
			return avgRes
		}

		for n, res := range resources {
			nodeSumRes[n] += res
			nodeRecordNum[n]++
		}
	}

	for n, res := range nodeSumRes {
		if res > 0 {
			avgRes[n] = nodeSumRes[n] / nodeRecordNum[n]
		} else {
			avgRes[n] = 0.0
		}
	}
	return avgRes
}

// CalculateJobNodeMaxResource get the max of particular resources of each PS
func CalculateJobNodeMaxResource(runtimeInfos []*common.JobRuntimeInfo, sampleStep int, resType string) map[uint64]float64 {
	nRecord := len(runtimeInfos)
	if sampleStep > nRecord {
		sampleStep = nRecord
	}
	maxRes := make(map[uint64]float64)
	for i := 0; i < sampleStep; i++ {
		rt := runtimeInfos[nRecord-i-1]
		res := getResourceAccordingToType(rt, resType)
		if res == nil {
			return nil
		}

		for n, r := range res {
			if r > maxRes[n] {
				maxRes[n] = r
			}
		}
	}
	return maxRes
}

func getResourceAccordingToType(rt *common.JobRuntimeInfo, resType string) map[uint64]float64 {
	var resources map[uint64]float64

	switch resType {
	case optimplcomm.ResourceTypePSCPU:
		resources = rt.PSCPU
	case optimplcomm.ResourceTypePSMemory:
		resources = rt.PSMemory
	case optimplcomm.ResourceTypeWorkerCPU:
		resources = rt.WorkerCPU
	case optimplcomm.ResourceTypeWorkerMemory:
		resources = rt.WorkerMemory
	default:
		log.Errorf("invalid resource type %s", resType)
		resources = nil
	}
	return resources
}

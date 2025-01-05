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
	"fmt"
	log "github.com/golang/glog"
	"github.com/intelligent-machine-learning/easydl/brain/pkg/common"
	"strconv"
	"strings"
)

// ExtractPodTypeAndIDFromName extracts pod type and id from the pod name
func ExtractPodTypeAndIDFromName(name string) (string, int) {
	workerSubstr := fmt.Sprintf("-%s-", common.WorkerTaskGroupName)
	psSubstr := fmt.Sprintf("-%s-", common.PSTaskGroupName)

	var ss []string
	var podType string
	if strings.Contains(name, workerSubstr) {
		podType = common.WorkerTaskGroupName
		ss = strings.Split(name, workerSubstr)
	} else if strings.Contains(name, psSubstr) {
		podType = common.PSTaskGroupName
		ss = strings.Split(name, psSubstr)
	}
	if len(ss) < 2 {
		log.Errorf("invalid pod name %s", name)
		return "", -1
	}
	idStr := ss[len(ss)-1]
	id, err := strconv.Atoi(idStr)
	if err != nil {
		log.Errorf("fail to convert pod id from %s: %v", idStr, err)
		return "", -1
	}
	return podType, id
}

// Decimal saves two decimal places
func Decimal(value float64) float64 {
	value, _ = strconv.ParseFloat(fmt.Sprintf("%.2f", value), 64)
	return value
}

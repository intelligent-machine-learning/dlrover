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

package master

import (
	elasticv1alpha1 "github.com/intelligent-machine-learning/easydl/dlrover/go/operator/api/v1alpha1"
	"github.com/stretchr/testify/assert"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"testing"
)

func TestCreateMasterPod(t *testing.T) {
	job := &elasticv1alpha1.ElasticJob{
		ObjectMeta: metav1.ObjectMeta{
			Name:        "test-ps",
			Namespace:   "easydl",
			Annotations: map[string]string{},
			Labels:      map[string]string{},
		},
	}

	manager := newManager()
	pod := manager.newEasydlMaster(job, initMasterIndex)
	assert.Equal(t, pod.Name, "test-ps-easydl-master-0")
}

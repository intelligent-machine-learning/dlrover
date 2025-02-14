// Copyright 2025 The DLRover Authors. All rights reserved.
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

package batchscheduler

import (
	"fmt"

	elasticjobv1 "github.com/intelligent-machine-learning/dlrover/go/elasticjob/api/v1alpha1"
	commonv1 "github.com/intelligent-machine-learning/dlrover/go/elasticjob/pkg/common/api/v1"
	"github.com/intelligent-machine-learning/dlrover/go/master/pkg/common"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	corev1 "k8s.io/api/core/v1"
)

var _ = Describe("Elastic", func() {
	It("Do scheduling to launch pods.", func() {
		jobContext := &common.JobContext{
			NameSpace:  "dlrover",
			Name:       "train-demo",
			MasterHost: "127.0.0.1",
			MasterPort: 12345,
		}

		container := corev1.Container{
			Name:            "main",
			Image:           "python:3.12.8",
			ImagePullPolicy: corev1.PullIfNotPresent,
			Command:         []string{"/bin/bash", "-c", "echo 0"},
		}
		replicas := make(map[commonv1.ReplicaType]*commonv1.ReplicaSpec)
		replicas["worker"] = &commonv1.ReplicaSpec{
			Replicas: 3,
			Template: corev1.PodTemplateSpec{
				Spec: corev1.PodSpec{
					Containers:    []corev1.Container{container},
					RestartPolicy: corev1.RestartPolicyNever,
				},
			},
		}
		schedulingPlan := &SchedulingPlan{
			ReplicaSpecs: replicas,
			OwnerJob:     &elasticjobv1.ElasticJob{},
		}
		scheduler := NewElasticScheduler()
		scheduler.DoScheduling(jobContext, schedulingPlan)
		Expect(scheduler.toCreatePods.Len()).To(Equal(3))
		for i := 0; i < 3; i++ {
			pod := scheduler.toCreatePods.PopFront().(*corev1.Pod)
			expectPodName := fmt.Sprintf("train-demo-worker-%d", i)
			Expect(pod.ObjectMeta.Name).To(Equal(expectPodName))
		}
	})
})

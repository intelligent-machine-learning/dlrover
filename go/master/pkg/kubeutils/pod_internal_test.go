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

package kubeutils

import (
	"errors"
	"fmt"
	"os"

	"github.com/intelligent-machine-learning/dlrover/go/master/pkg/common"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	corev1 "k8s.io/api/core/v1"
	kubeerrors "k8s.io/apimachinery/pkg/api/errors"
)

var _ = Describe("Pod", func() {
	It("Create a Pod", func() {
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
		podConfig := &PodConfig{
			Replica: &ReplicaConfig{
				Type:   "worker",
				ID:     0,
				Number: 8,
				Rank:   0,
			},
			TemplateSpec: &corev1.PodTemplateSpec{
				Spec: corev1.PodSpec{
					Containers:    []corev1.Container{container},
					RestartPolicy: corev1.RestartPolicyNever,
				},
			},
		}
		pod := BuildPod(jobContext, podConfig)
		Expect(pod.ObjectMeta.Name).To(Equal("train-demo-worker-0"))
		Expect(pod.ObjectMeta.Namespace).To(Equal("dlrover"))
		jobName, ok := pod.ObjectMeta.Labels[labelJobKey]
		Expect(ok).To(BeTrue())
		Expect(jobName).To(Equal("train-demo"))
		replicaType, ok := pod.ObjectMeta.Labels[labelReplicaTypeKey]
		Expect(ok).To(BeTrue())
		Expect(replicaType).To(Equal("worker"))

		configPath := os.Getenv("KUBERNETES_CONFIG_PATH")
		if _, err := os.Stat(configPath); errors.Is(err, os.ErrNotExist) {
			Skip(fmt.Sprintf("The config file %s is not exist.", configPath))
		}

		k8sClient := NewK8sClient(configPath)
		pod.ObjectMeta.Namespace = "no-namspace"
		err := k8sClient.CreatePod("dlrover", pod)
		Expect(kubeerrors.IsBadRequest(err)).To(BeTrue())

		pod.ObjectMeta.Namespace = "dlrover"
		err = k8sClient.CreatePod("dlrover", pod)
		Expect(kubeerrors.IsAlreadyExists(err)).To(BeTrue())

		pod.ObjectMeta.Name = ""
		err = k8sClient.CreatePod("dlrover", pod)
		Expect(kubeerrors.IsInvalid(err)).To(BeTrue())
	})
})

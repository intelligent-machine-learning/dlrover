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

package kubernetes

import (
	elasticjob "github.com/intelligent-machine-learning/dlrover/go/elasticjob/api/v1alpha1"

	"k8s.io/apimachinery/pkg/runtime"
)

const (
	GROUP   = "elastic.iml.github.io"
	VERSION = "v1alpha1"
)

func GetElasticJobInstance(client *K8sClient, namespace string, jobName string) *elasticjob.ElasticJob {

	gvr := GetGroupVersionResource(GROUP, VERSION, "elasticjobs")
	utd, err := client.GetCustomResourceInstance(namespace, jobName, gvr)
	if err != nil {
		return nil
	}
	// Unstructured -> job
	var job elasticjob.ElasticJob
	runtime.DefaultUnstructuredConverter.FromUnstructured(utd.UnstructuredContent(), &job)
	return &job
}

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

package main

import (
	"flag"
	"strconv"

	logger "github.com/sirupsen/logrus"
	"k8s.io/client-go/kubernetes"

	master "github.com/intelligent-machine-learning/dlrover/go/master/pkg"
	"github.com/intelligent-machine-learning/dlrover/go/master/pkg/kubernetes"
	"github.com/intelligent-machine-learning/dlrover/go/master/pkg/server"
)

func main() {
	var k8sScheduling bool
	var namespace string
	var jobName string
	var port int

	flag.BoolVar(&k8sScheduling, "k8s_scheduling", true, "Whether to enable scheduling pods on k8s.")
	flag.StringVar(&namespace, "namespace", "default", "The name of the Kubernetes namespace.")
	flag.StringVar(&jobName, "job_name", "", "The dlrover/elasticjob name.")
	flag.IntVar(&port, "port", 8080, "The port which the master service binds to.")

	router := server.NewRouter()
	go router.Run(":" + strconv.Itoa(port))

	// Listen and serve on defined port
	logger.Infof("The master starts with namespece %s, jobName %s, port %d", namespace, jobName, port)
	var k8sClient *kubernetes.K8sClient
	if k8sScheduling {
		k8sClient = kubernetes.NewK8sClient(namespace, jobName)
	}
	master := master.NewJobMaster(namespace, jobName, k8sClient)
	master.Run()
}
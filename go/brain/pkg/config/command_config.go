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

package config

import "flag"

const (
	// SpecKubeConfigName is the spec name of kube config
	SpecKubeConfigName = "kubeConfig"
	// SpecMasterName is the spec name of master name
	SpecMasterName = "apiServer"
	// SpecServiceConfigMapName is the spec name of service config map name
	SpecServiceConfigMapName = "serviceConfigMapName"
	// SpecServiceConfigMapKey is the spec name of service config map key
	SpecServiceConfigMapKey = "serviceConfigMapKey"
	// SpecNamespaceName is the spec name of namespace
	SpecNamespaceName = "namespace"
	// SpecServerPort is the port of the EasyDL server gRPC service
	SpecServerPort = "port"
)

// Spec is the struct of configure specifications
type Spec struct {
	// ApiServer is the api server
	APIServer string
	// KubeConfig is the kube config
	KubeConfig string
	// ServiceConfigMapName is the service config map name
	ServiceConfigMapName string
	// ServiceConfigMapKey is the service config map key
	ServiceConfigMapKey string
	// Namespace is the namespace
	Namespace string
	// Port is the port of the EasyDL server gRPC service
	Port string
}

// CommandConfig is the variable of type ConfigSpec
var CommandConfig = &Spec{}

func init() {
	flag.StringVar(&CommandConfig.KubeConfig,
		SpecKubeConfigName,
		"",
		"Path to a kubeConfig. Only required if out-of-cluster.")
	flag.StringVar(&CommandConfig.APIServer,
		SpecMasterName,
		"",
		"The address of the Kubernetes API server. Overrides any value in kubeConfig. Only required if out-of-cluster.")
	flag.StringVar(&CommandConfig.ServiceConfigMapName,
		SpecServiceConfigMapName,
		"",
		"The name of k8s config map")
	flag.StringVar(&CommandConfig.ServiceConfigMapKey,
		SpecServiceConfigMapKey,
		"",
		"The key of k8s config map")
	flag.StringVar(&CommandConfig.Namespace,
		SpecNamespaceName,
		"",
		"The namespace name used for k8s ConfigMap & LeaderElection")
	flag.StringVar(
		&CommandConfig.Port,
		SpecServerPort,
		":50001",
		"Port of the EasyDL server gRPC service")
}

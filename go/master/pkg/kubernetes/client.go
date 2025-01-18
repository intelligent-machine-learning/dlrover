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
	"context"

	logger "github.com/sirupsen/logrus"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/dynamic"
	k8sApi "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"
)

type K8sClient struct {
	config        *rest.Config
	clientset     *k8sApi.Clientset
	dynamicClient *dynamic.DynamicClient
}

func NewK8sClient(kubeConfigPath string) *K8sClient {
	client := &K8sClient{}

	// creates the in-cluster config
	if kubeConfigPath == "" {
		config, err := rest.InClusterConfig()
		if err != nil {
			panic(err.Error())
		}
		client.config = config
	} else {
		config, err := clientcmd.BuildConfigFromFlags("", kubeConfigPath)
		if err != nil {
			panic(err.Error())
		}
		client.config = config
	}

	// creates the clientset
	clientset, err := k8sApi.NewForConfig(client.config)
	if err != nil {
		panic(err.Error())
	}
	client.clientset = clientset

	dynamicClient, err := dynamic.NewForConfig(client.config)
	if err != nil {
		panic(err.Error())
	}
	client.dynamicClient = dynamicClient
	return client
}

func (client *K8sClient) GetCustomResourceInstance(
	namespace string, name string, gvr schema.GroupVersionResource) (
	*unstructured.Unstructured, error,
) {
	// Unstructured
	utd, err := client.dynamicClient.
		Resource(gvr).
		Namespace(namespace).
		Get(context.TODO(), name, metav1.GetOptions{})
	if err != nil {
		logger.Infof("fail to get %s %s", gvr.String(), name)
	}
	return utd, err
}

// GetGroupVersionResource :- gets GroupVersionResource for dynamic client
func GetGroupVersionResource(group, version, resource string) schema.GroupVersionResource {
	return schema.GroupVersionResource{Group: group, Version: version, Resource: resource}
}

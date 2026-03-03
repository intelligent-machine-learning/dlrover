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
	"context"

	logger "github.com/sirupsen/logrus"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/dynamic"
	k8sApi "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"
	"k8s.io/utils/ptr"
)

// GlobalK8sClient is the global client to access a k8s cluster.
var GlobalK8sClient *K8sClient

// K8sClient contains the instance to access a k8s cluster.
type K8sClient struct {
	namespace     string
	config        *rest.Config
	clientset     *k8sApi.Clientset
	dynamicClient *dynamic.DynamicClient
}

// GetGroupVersionResource :- gets GroupVersionResource for dynamic client
func GetGroupVersionResource(group, version, resource string) schema.GroupVersionResource {
	return schema.GroupVersionResource{Group: group, Version: version, Resource: resource}
}

// NewGlobalK8sClient initialize the global k8s client.
func NewGlobalK8sClient(kubeConfigPath string, namespace string) {
	GlobalK8sClient = NewK8sClient(kubeConfigPath, namespace)
}

// NewK8sClient creates a k8s client instance.
func NewK8sClient(kubeConfigPath string, namespace string) *K8sClient {
	client := &K8sClient{
		namespace: namespace,
	}

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

// GetCustomResourceInstance gets a custom resource instance from a k8s cluster.
func (client *K8sClient) GetCustomResourceInstance(name string, gvr schema.GroupVersionResource) (
	*unstructured.Unstructured, error,
) {
	// Unstructured
	utd, err := client.dynamicClient.
		Resource(gvr).
		Namespace(client.namespace).
		Get(context.Background(), name, metav1.GetOptions{})
	if err != nil {
		logger.Infof("fail to get %s %s", gvr.String(), name)
	}
	return utd, err
}

// CreatePod creates a Pod instance in the cluster
func (client *K8sClient) CreatePod(ctx context.Context, pod *corev1.Pod) error {
	_, err := client.clientset.
		CoreV1().
		Pods(client.namespace).
		Create(ctx, pod, metav1.CreateOptions{})
	return err
}

// RemovePod removes a Pod instance in the cluster
func (client *K8sClient) RemovePod(name string) error {
	err := client.clientset.CoreV1().Pods(client.namespace).Delete(
		context.Background(),
		name,
		metav1.DeleteOptions{GracePeriodSeconds: ptr.To(int64(0))},
	)
	return err
}

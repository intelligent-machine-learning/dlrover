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
)

type K8sClient struct {
	config        *rest.Config
	clientset     *k8sApi.Clientset
	dynamicClient *dynamic.DynamicClient
}

func NewK8sClient() *K8sClient {
	client := &K8sClient{}

	// creates the in-cluster config
	config, err := rest.InClusterConfig()
	if err != nil {
		panic(err.Error())
	}
	client.config = config

	// creates the clientset
	client.clientset, err = k8sApi.NewForConfig(config)
	if err != nil {
		panic(err.Error())
	}

	client.dynamicClient, err = dynamic.NewForConfig(config)
	if err != nil {
		panic(err.Error())
	}
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

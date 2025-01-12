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

package kubernetes

import (
	"os"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

var _ = Describe("Elasticjob", func() {
	It("Get an elasticjob instance", func() {
		kubeConfigPath := os.Getenv("KUBENETES_CLUSTER_CONFIG")
		if kubeConfigPath != "" {
			k8sClient := NewK8sClient(kubeConfigPath)
			job := GetElasticJobInstance(k8sClient, "dlrover", "torch-mnist")
			Expect(job.Name).To(Equal("torch-minst"))
		}
	})
})

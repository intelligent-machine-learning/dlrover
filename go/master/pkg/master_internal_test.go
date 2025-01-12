package master

import (
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

var _ = Describe("Master", func() {
	It("Create a master", func() {
		master := NewJobMaster("dlrover", "test-master", false)
		Expect(master.Namespace).To(Equal("dlrover"))
		Expect(master.JobName).To(Equal("test-master"))
	})
})

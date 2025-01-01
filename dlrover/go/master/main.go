package main

import (
	"flag"
	"strconv"

	logger "github.com/sirupsen/logrus"

	"github.com/intelligent-machine-learning/dlrover/go/master/pkg/server"
)

func main() {
	var namespace string
	var jobName string
	var port int

	flag.StringVar(&namespace, "namespace", "default", "The name of the Kubernetes namespace.")
	flag.StringVar(&jobName, "job_name", "", "The dlrover/elasticjob name.")
	flag.IntVar(&port, "port", 8080, "The port which the master service binds to.")
	router := server.NewRouter()

	// Listen and serve on defined port
	logger.Infof("The master starts with namespece %s, jobName %s, port %d", namespace, jobName, port)
	router.Run(":" + strconv.Itoa(port))
}

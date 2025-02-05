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

package common

import (
	"fmt"
	"net"
	"os"
	"strconv"
	"time"
)

const masterServicePort = 215000

// JobContext stores the elastic job context.
type JobContext struct {
	// Namespace is the kubernetes namespace where the job runs.
	NameSpace string
	// Name is the name of an elastic job.
	Name string
	// MasterHost is the host of master service.
	MasterHost string
	// MasterPort is the host of master port.
	MasterPort int
}

// NewJobContext creates a job context.
func NewJobContext(namespace string, name string) *JobContext {
	host := fmt.Sprintf("elasticjob-%s-dlrover-master", name)
	port := masterServicePort

	if !checkAddressReachable(host, port) {
		host = os.Getenv("MY_POD_IP")
		freePort, err := getFreePort()
		if err != nil {
			panic(err.Error())
		}
		port = freePort
	}

	return &JobContext{
		NameSpace:  namespace,
		Name:       name,
		MasterHost: host,
		MasterPort: port,
	}
}

func checkAddressReachable(host string, port int) bool {
	timeout := time.Second
	masterAddr := net.JoinHostPort(host, strconv.Itoa(port))
	conn, err := net.DialTimeout("tcp", masterAddr, timeout)
	if err != nil {
		return false
	}
	if conn != nil {
		defer conn.Close()
	}
	return true
}

func getFreePort() (int, error) {
	addr, err := net.ResolveTCPAddr("tcp", "localhost:0")
	if err != nil {
		return 0, err
	}

	l, err := net.ListenTCP("tcp", addr)
	if err != nil {
		return 0, err
	}

	defer l.Close()
	return l.Addr().(*net.TCPAddr).Port, nil
}

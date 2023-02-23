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

package main

import (
	"context"
	"flag"
	log "github.com/golang/glog"
	"github.com/intelligent-machine-learning/easydl/brain/pkg/common"
	"github.com/intelligent-machine-learning/easydl/brain/pkg/config"
	pb "github.com/intelligent-machine-learning/easydl/brain/pkg/proto"
	"github.com/intelligent-machine-learning/easydl/brain/pkg/server"
	"google.golang.org/grpc"
	kubeclientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/clientcmd"
	"net"
)

func main() {
	log.Info("Start DLRover Brain")
	flag.Parse()
	mConfig := config.CommandConfig

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	cfg, err := clientcmd.BuildConfigFromFlags(mConfig.APIServer, mConfig.KubeConfig)
	if err != nil {
		log.Fatalf("Error building kube config: %v", err)
	}

	kubeClient, err := kubeclientset.NewForConfig(cfg)
	if err != nil {
		log.Fatalf("Error building kube client: %v", err.Error())
	}

	// Start error handler
	log.Info("Start error handler")
	errHandler, err := common.NewStopAllErrorHandler(cancel)
	if err != nil {
		log.Fatalf("Create ErrorHandler error: %v", err)
	} else {
		go errHandler.HandleError(ctx)
	}

	log.Infof("namespace=%s, serviceConfigMapName=%s, serviceConfigMapKey=%s", mConfig.Namespace,
		mConfig.ServiceConfigMapName, mConfig.ServiceConfigMapKey)
	conf := config.NewEmptyConfig()
	conf.Set(config.KubeClientInterface, kubeClient)
	conf.Set(config.BrainServerConfigMapName, mConfig.ServiceConfigMapName)
	conf.Set(config.BrainServerConfigMapKey, mConfig.ServiceConfigMapKey)
	conf.Set(config.Namespace, mConfig.Namespace)

	lis, err := net.Listen("tcp", mConfig.Port)
	if err != nil {
		log.Fatalf("failed to listen: %v", err)
	}
	s := grpc.NewServer()

	brainServer, err := server.NewBrainServer(conf)
	if err != nil {
		log.Fatalf("fail to create Brain server: %v", err)
	}

	err = brainServer.Run(ctx, errHandler)
	if err != nil {
		log.Fatalf("Fail to run Brain server: %v", err)
	}

	pb.RegisterBrainServer(s, brainServer)
	log.Infof("server listening at %v", lis.Addr())
	if err = s.Serve(lis); err != nil {
		log.Fatalf("failed to serve: %v", err)
	}
}

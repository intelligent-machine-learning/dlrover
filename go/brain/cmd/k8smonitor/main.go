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
	k8swatcher "github.com/intelligent-machine-learning/easydl/brain/pkg/platform/k8s/watcher"
	kubeclientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/clientcmd"
)

func main() {
	log.Info("Start DLRover K8s monitor")
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
	conf.Set(config.KubeWatcherConfigMapName, mConfig.ServiceConfigMapName)
	conf.Set(config.KubeWatcherConfigMapKey, mConfig.ServiceConfigMapKey)
	conf.Set(config.Namespace, mConfig.Namespace)

	manager, err := k8swatcher.NewManager(conf)
	if err != nil {
		log.Fatalf("fail to create brain processor server: %v", err)
	}

	log.Infof("create kube watcher manager: %+v", manager)

	err = manager.Run(ctx, errHandler, errHandler)
	if err != nil {
		log.Errorf("fail to run the kube watcher manager: %v", err)
	}

	<-ctx.Done()
}

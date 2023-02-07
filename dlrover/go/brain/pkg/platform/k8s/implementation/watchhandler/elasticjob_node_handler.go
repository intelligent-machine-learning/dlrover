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

package watchhandler

import (
	log "github.com/golang/glog"
	"github.com/intelligent-machine-learning/easydl/brain/pkg/common"
	"github.com/intelligent-machine-learning/easydl/brain/pkg/config"
	datastoreapi "github.com/intelligent-machine-learning/easydl/brain/pkg/datastore/api"
	"github.com/intelligent-machine-learning/easydl/brain/pkg/datastore/recorder/mysql"
	handlerutils "github.com/intelligent-machine-learning/easydl/brain/pkg/platform/k8s/implementation/watchhandler/utils"
	watchercommon "github.com/intelligent-machine-learning/easydl/brain/pkg/platform/k8s/watcher/common"
	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"time"
)

const (
	elasticJobNodeHandlerName = "elasticjob_node_handler"
)

func init() {
	registerWatchHandlerFunc(elasticJobNodeHandlerName, registerElasticJobNodeHandler)
}

// ElasticJobNodeHandler is to watch and record the status of elastic job nodes
type ElasticJobNodeHandler struct {
	name      string
	conf      *config.Config
	dataStore datastoreapi.DataStore
}

func newElasticJobNodeHandler(name string, conf *config.Config, dataStore datastoreapi.DataStore) (watchercommon.EventHandler, error) {
	return &ElasticJobNodeHandler{
		name:      name,
		conf:      conf,
		dataStore: dataStore,
	}, nil
}

func registerElasticJobNodeHandler(kubeWatcher *watchercommon.KubeWatcher, name string, conf *config.Config, dataStore datastoreapi.DataStore) error {
	handler, err := newElasticJobNodeHandler(name, conf, dataStore)
	if err != nil {
		return err
	}
	filterEnqueueRequestForObject := watchercommon.NewWatchFilterEnqueueRequestForObject(handlerutils.ElasticJobNodeFilterFunc)
	if err = kubeWatcher.WatchKubeResource(v1.SchemeGroupVersion.WithKind("Pod"), handler,
		watchercommon.WithEnqueueEventHandler(filterEnqueueRequestForObject), watchercommon.WithMaxConcurrent(workerConcurrent)); err != nil {
		return err
	}
	return nil
}

// HandleCreateEvent handles create events
func (handler *ElasticJobNodeHandler) HandleCreateEvent(object runtime.Object, event watchercommon.Event) error {
	node := &v1.Pod{}
	unstructObj := object.(*unstructured.Unstructured)

	runtime.DefaultUnstructuredConverter.FromUnstructured(unstructObj.Object, node)

	log.Infof("Job node %s is created", node.Name)

	record := &mysql.JobNode{
		UID:       string(node.UID),
		Name:      node.Name,
		CreatedAt: time.Now(),
	}

	cond := &datastoreapi.Condition{
		Type: common.TypeUpsertJobNode,
	}
	return handler.dataStore.PersistData(cond, record, nil)
}

// HandleUpdateEvent handles update events
func (handler *ElasticJobNodeHandler) HandleUpdateEvent(object runtime.Object, oldObject runtime.Object, event watchercommon.Event) error {
	return nil
}

// HandleDeleteEvent handles delete events
func (handler *ElasticJobNodeHandler) HandleDeleteEvent(deleteObject runtime.Object, event watchercommon.Event) error {
	return nil
}

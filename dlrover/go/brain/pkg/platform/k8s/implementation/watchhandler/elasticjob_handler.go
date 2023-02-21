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
	k8scommon "github.com/intelligent-machine-learning/easydl/brain/pkg/platform/k8s/common"
	handlerutils "github.com/intelligent-machine-learning/easydl/brain/pkg/platform/k8s/implementation/watchhandler/utils"
	watchercommon "github.com/intelligent-machine-learning/easydl/brain/pkg/platform/k8s/watcher/common"
	elasticv1alpha1 "github.com/intelligent-machine-learning/easydl/dlrover/go/operator/api/v1alpha1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"time"
)

const (
	elasticJobHandlerName = "elastic_job_handler"
)

func init() {
	registerWatchHandlerFunc(elasticJobHandlerName, registerElasticJobHandler)
}

// ElasticJobHandler is to watch and record the status of elastic jobs
type ElasticJobHandler struct {
	name      string
	conf      *config.Config
	dataStore datastoreapi.DataStore
}

func newElasticJobHandler(name string, conf *config.Config, dataStore datastoreapi.DataStore) (watchercommon.EventHandler, error) {
	return &ElasticJobHandler{
		name:      name,
		conf:      conf,
		dataStore: dataStore,
	}, nil
}

func registerElasticJobHandler(kubeWatcher *watchercommon.KubeWatcher, name string, conf *config.Config, dataStore datastoreapi.DataStore) error {
	handler, err := newElasticJobHandler(name, conf, dataStore)
	if err != nil {
		return err
	}
	filterEnqueueRequestForObject := watchercommon.NewWatchFilterEnqueueRequestForObject(handlerutils.ElasticJobFilterFunc)
	if err = kubeWatcher.WatchKubeResource(elasticv1alpha1.SchemeGroupVersionKind, handler,
		watchercommon.WithEnqueueEventHandler(filterEnqueueRequestForObject), watchercommon.WithMaxConcurrent(workerConcurrent)); err != nil {
		return err
	}
	return nil
}

// HandleCreateEvent handles create events
func (handler *ElasticJobHandler) HandleCreateEvent(object runtime.Object, event watchercommon.Event) error {
	job := &elasticv1alpha1.ElasticJob{}
	unstructObj := object.(*unstructured.Unstructured)

	runtime.DefaultUnstructuredConverter.FromUnstructured(unstructObj.Object, job)
	log.Infof("job %s is created", job.Name)

	record := &mysql.Job{
		UID:       string(job.UID),
		Name:      job.Name,
		CreatedAt: time.Now(),
	}

	cond := &datastoreapi.Condition{
		Type: common.TypeUpsertJob,
	}
	return handler.dataStore.PersistData(cond, record, nil)
}

// HandleUpdateEvent handles update events
func (handler *ElasticJobHandler) HandleUpdateEvent(object runtime.Object, oldObject runtime.Object, event watchercommon.Event) error {
	newJob := &elasticv1alpha1.ElasticJob{}
	oldJob := &elasticv1alpha1.ElasticJob{}

	unstructNewObj := object.(*unstructured.Unstructured)
	unstructOldObj := oldObject.(*unstructured.Unstructured)

	runtime.DefaultUnstructuredConverter.FromUnstructured(unstructOldObj.Object, oldJob)
	runtime.DefaultUnstructuredConverter.FromUnstructured(unstructNewObj.Object, newJob)

	jobUID := string(newJob.UID)

	cond := &datastoreapi.Condition{
		Type: common.TypeGetDataGetJob,
		Extra: &mysql.JobCondition{
			UID: jobUID,
		},
	}
	record := &mysql.Job{}

	err := handler.dataStore.GetData(cond, record)
	if err != nil {
		log.Errorf("fail to get job record for %s: %v", jobUID, err)
		return err
	}

	if newJob.Status.Phase == k8scommon.JobPhaseRunning && record.StartedAt.IsZero() {
		record.StartedAt = newJob.Status.StartTime.Time
	}

	if newJob.Status.Phase == k8scommon.JobPhaseSucceeded && record.FinishedAt.IsZero() {
		record.FinishedAt = newJob.Status.CompletionTime.Time
	}

	upsertCond := &datastoreapi.Condition{
		Type: common.TypeUpsertJob,
	}
	return handler.dataStore.PersistData(upsertCond, record, nil)
}

// HandleDeleteEvent handles delete events
func (handler *ElasticJobHandler) HandleDeleteEvent(deleteObject runtime.Object, event watchercommon.Event) error {
	return nil
}

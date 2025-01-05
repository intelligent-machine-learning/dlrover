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

package common

import (
	"context"
	log "github.com/golang/glog"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sync"
	"time"
)

// WatchRequest contains the information necessary to reconcile a Kubernetes object.  This includes the
// information to uniquely identify the object - its Name and Namespace.  It does NOT contain information about
// any specific Event or the object contents itself.
type WatchRequest struct {
	// NamespacedName is the name and namespace of the object to reconcile.
	types.NamespacedName

	// uuid of the kubernetes object
	UUID string
}

// Result contains the result of a Reconciler invocation.
type Result struct {
	// Requeue tells the Controller to requeue the reconcile key.  Defaults to false.
	Requeue bool

	// RequeueAfter if greater than 0, tells the Controller to requeue the reconcile key after the Duration.
	// Implies that Requeue is true, there is no need to set Requeue to true at the same time as RequeueAfter.
	RequeueAfter time.Duration
}

// Reconciler is the interface of kube reconsiler
type Reconciler interface {
	// Reconciler performs a full reconciliation for the object referred to by the WatchEvent.
	// The Controller will requeue the WatchEvent to be processed again if an error is non-nil or
	// Result.Requeue is true, otherwise upon completion it will remove the work from the queue.
	Reconcile(WatchRequest) (Result, error)
}

// WatchReconciler is used to distinguish the different events and execute corresponding EventHandler.
type WatchReconciler struct {
	client.Client
	Scheme      *runtime.Scheme
	Handler     EventHandler
	GVK         schema.GroupVersionKind
	cacheObject sync.Map
}

// NewWatchReconciler returns a WatchReconsiler
func NewWatchReconciler(GVK schema.GroupVersionKind, client client.Client, scheme *runtime.Scheme, handler EventHandler) *WatchReconciler {
	return &WatchReconciler{GVK: GVK, Client: client, Scheme: scheme, Handler: handler}
}

// Reconcile processes a WatchRequest
func (r *WatchReconciler) Reconcile(req WatchRequest) (Result, error) {
	obj := &unstructured.Unstructured{}
	obj.SetGroupVersionKind(r.GVK)

	//obj := &metav1.PartialObjectMetadata{}
	//obj.SetGroupVersionKind(r.GVK)

	//obj, err := r.Scheme.New(r.GVK)
	//if err != nil {
	//	log.Errorf("failed to new runtime scheme [GroupVersionKind:%s] with err:%v", r.GVK, err)
	//	return Result{}, err
	//}

	// get the cache workerInfos
	value, found := r.cacheObject.Load(req.UUID)
	if !found {
		// 1. EventCreate
		if err := r.Get(context.Background(), req.NamespacedName, obj); err != nil {
			if errors.IsNotFound(err) {
				log.Warning("object not found anymore, exit reconcile!")
				return Result{}, nil
			}
			log.Errorf("failed to get the object [uuid:%s] [GroupVersionKind:%s] from list-watch cache", req.UUID, r.GVK)
			return Result{}, err
		}

		log.Infof("WatchReconciler handle EventCreate: obj [uuid:%s] [GroupVersionKind:%s]", req.UUID, r.GVK)
		if err := r.Handler.HandleCreateEvent(obj, Event{
			NamespacedName: req.NamespacedName,
			UUID:           req.UUID,
			EventType:      EventCreate,
		}); err != nil {
			log.Errorf("failed to handle EventCreate: obj [uuid:%s] [GroupVersionKind:%s] with err:%v", req.UUID, r.GVK, err)
			return Result{}, err
		}
		r.cacheObject.Store(req.UUID, obj)
		return Result{}, nil
	}
	// 2. EventUpdate or EventDelete
	if err := r.Get(context.Background(), req.NamespacedName, obj); err != nil {
		if errors.IsNotFound(err) {
			// EventDelete
			log.Infof("WatchReconciler handle EventDelete: obj [uuid:%s] [GroupVersionKind:%s]", req.UUID, r.GVK)
			if err = r.Handler.HandleDeleteEvent(value.(runtime.Object), Event{
				NamespacedName: req.NamespacedName,
				UUID:           req.UUID,
				EventType:      EventDelete,
			}); err != nil {
				log.Errorf("Failed to handle EventDelete: obj [uuid:%s] [GroupVersionKind:%s] with err:%v", req.UUID, r.GVK, err)
				return Result{}, err
			}
			r.cacheObject.Delete(req.UUID)
			return Result{}, nil
		}
		// list-watch get wrong exception
		log.Errorf("Failed to get the object [uuid:%s] [GroupVersionKind:%s] from list-watch cache with err:%v", req.UUID, r.GVK, err)
		return Result{}, err
	}

	metaObj := obj
	if metaObj.GetDeletionTimestamp() != nil {
		// EventDelete
		log.Infof("WatchReconciler handle EventDelete: obj [uuid:%s] [GroupVersionKind:%s]", req.UUID, r.GVK)
		if err := r.Handler.HandleDeleteEvent(obj, Event{
			NamespacedName: req.NamespacedName,
			UUID:           req.UUID,
			EventType:      EventDelete,
		}); err != nil {
			log.Errorf("Failed to handle EventDelete: obj [uuid:%s] [GroupVersionKind:%s] with err:%v", req.UUID, r.GVK, err)
			return Result{}, err
		}
		r.cacheObject.Store(req.UUID, obj)
		return Result{}, nil
	}

	// EventUpdate
	oldObj := value.(runtime.Object)
	log.Infof("WatchReconciler handle EventUpdate: obj [uuid:%s] [GroupVersionKind:%s]", req.UUID, r.GVK)
	if err := r.Handler.HandleUpdateEvent(obj, oldObj, Event{
		NamespacedName: req.NamespacedName,
		UUID:           req.UUID,
		EventType:      EventUpdate,
	}); err != nil {
		log.Errorf("Failed to handle EventUpdate: obj [uuid:%s] [GroupVersionKind:%s] with err:%v", req.UUID, r.GVK, err)
		return Result{}, err
	}
	r.cacheObject.Store(req.UUID, obj)
	return Result{}, nil
}

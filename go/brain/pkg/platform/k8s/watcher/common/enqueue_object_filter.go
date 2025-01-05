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
	log "github.com/golang/glog"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/util/workqueue"
	"sigs.k8s.io/controller-runtime/pkg/event"
	"sigs.k8s.io/controller-runtime/pkg/handler"
)

var _ handler.EventHandler = &WatchFilterEnqueueRequestForObject{}

// FilterEnqueueRequestFun is the function to filter events
type FilterEnqueueRequestFun func(meta metav1.Object) bool

// WatchFilterEnqueueRequestForObject enqueues a WatchRequest containing the Name 、Namespace and UUID of the object that is the source of the Event.
// (e.g. the created / deleted / updated objects Name and Namespace).  reconcile.WatchFilterEnqueueRequestForObject is used by almost all
// Controllers that have associated Resources (e.g. CRDs) to reconcile the associated Resource.
type WatchFilterEnqueueRequestForObject struct {
	filterFun FilterEnqueueRequestFun
}

// NewWatchFilterEnqueueRequestForObject returns a new watch filter enqueue objects
func NewWatchFilterEnqueueRequestForObject(filterFun FilterEnqueueRequestFun) *WatchFilterEnqueueRequestForObject {
	return &WatchFilterEnqueueRequestForObject{filterFun: filterFun}
}

// Create implements EventHandler
func (e *WatchFilterEnqueueRequestForObject) Create(evt event.CreateEvent, q workqueue.RateLimitingInterface) {
	if evt.Object == nil {
		log.Errorf("CreateEvent received invalid event: %v", evt)
		return
	}
	if e.filterFun(evt.Object) {
		q.Add(WatchRequest{
			NamespacedName: types.NamespacedName{Name: evt.Object.GetName(), Namespace: evt.Object.GetNamespace()}, UUID: string(evt.Object.GetUID()),
		})
	}
}

// Update implements EventHandler
func (e *WatchFilterEnqueueRequestForObject) Update(evt event.UpdateEvent, q workqueue.RateLimitingInterface) {
	if evt.ObjectOld != nil {
		if e.filterFun(evt.ObjectOld) {
			q.Add(WatchRequest{
				NamespacedName: types.NamespacedName{Name: evt.ObjectOld.GetName(), Namespace: evt.ObjectOld.GetNamespace()}, UUID: string(evt.ObjectOld.GetUID()),
			})
		}
	} else {
		log.Errorf("UpdateEvent received invalid event: %v", evt)
	}

	if evt.ObjectNew != nil {
		if e.filterFun(evt.ObjectNew) {
			q.Add(WatchRequest{
				NamespacedName: types.NamespacedName{Name: evt.ObjectNew.GetName(), Namespace: evt.ObjectNew.GetNamespace()}, UUID: string(evt.ObjectNew.GetUID()),
			})
		}
	} else {
		log.Errorf("UpdateEvent received event with no new metadata: %v", evt)
	}
}

// Delete implements EventHandler
func (e *WatchFilterEnqueueRequestForObject) Delete(evt event.DeleteEvent, q workqueue.RateLimitingInterface) {
	if evt.Object == nil {
		log.Errorf("DeleteEvent received event %v with no metadata", evt)
		return
	}
	if e.filterFun(evt.Object) {
		q.Add(WatchRequest{
			NamespacedName: types.NamespacedName{Name: evt.Object.GetName(), Namespace: evt.Object.GetNamespace()}, UUID: string(evt.Object.GetUID()),
		})
	}
}

// Generic implements EventHandler
func (e *WatchFilterEnqueueRequestForObject) Generic(evt event.GenericEvent, q workqueue.RateLimitingInterface) {
	if evt.Object == nil {
		log.Errorf("GenericEvent received event with no metadata: %v", evt)
		return
	}
	if e.filterFun(evt.Object) {
		q.Add(WatchRequest{
			NamespacedName: types.NamespacedName{Name: evt.Object.GetName(), Namespace: evt.Object.GetNamespace()}, UUID: string(evt.Object.GetUID()),
		})
	}
}

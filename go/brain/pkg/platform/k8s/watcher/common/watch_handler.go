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
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
)

// EventType contains the watch events of different Kubernetes resources
type EventType string

const (
	// EventCreate is a create event - e.g. Pod Creation.
	EventCreate EventType = "created"

	// EventUpdate is a update event - e.g. Pod Updated.
	EventUpdate EventType = "updated"

	// EventDelete is a delete event - e.g. Pod Deleted.
	EventDelete EventType = "deleted"
)

// Event contains the information necessary to watch a Kubernetes object.  This includes the
// information to uniquely identify the object - its Name and Namespace. This includes type of Event
// contains (EventCreate、EventUpdate、EventDelete、EventGeneric)。 This also includes the uuid of the kubernetes object.
type Event struct {
	// OldNamespacedName is the name and namespace of the object to watch.
	NamespacedName types.NamespacedName

	// uuid of the kubernetes object
	UUID string

	// type of Event
	EventType EventType
}

// EventHandler is an interface that used to process generic event.
// It handles delete events、update events、create events (including the EventCreate、EventUpdate、EventDelete).
type EventHandler interface {
	// HandleCreateEvent can be called to handle create events of specific kubernetes resource
	// Note: When the kube-watch starts for the first time, it will execute a list-cache synchronization with kubernetes apiServer.
	// At this time, all events will be treated as a create event.
	HandleCreateEvent(object runtime.Object, event Event) error
	// HandleUpdateEvent can be called to handle update events of specific kubernetes resource.
	// Note: Object is the object after the update event happens, oldObject is the object before the update event happens.
	HandleUpdateEvent(object runtime.Object, oldObject runtime.Object, event Event) error
	// HandleDeleteEvent can be called to handle delete events of specific kubernetes resource
	// Note: When the CRD is being deleted, it may call HandleEvent multiple times. Eg. Pod Deleted.
	// Therefore, the implementation of the HandleDeleteEvent interface can be able to be called multiple times
	HandleDeleteEvent(deleteObject runtime.Object, event Event) error
}

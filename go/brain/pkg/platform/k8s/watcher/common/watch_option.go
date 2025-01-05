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
	"sigs.k8s.io/controller-runtime/pkg/handler"
	"sigs.k8s.io/controller-runtime/pkg/manager"
	"sigs.k8s.io/controller-runtime/pkg/predicate"
)

// KubeWatchOptions is the struct of options of KubeWatcher
type KubeWatchOptions struct {
	// Options are the arguments for creating a new controller Manager
	manager.Options
	// QPS indicates the maximum QPS to the master from this client.
	// If it's zero, the created RESTClient will use DefaultQPS: 5
	QPS float32
	// Burst is the maximum burst for throttle.
	// If it's zero, the created RESTClient will use DefaultBurst: 10.
	Burst int
	// UserAgent is an optional field that specifies the caller of this request.
	UserAgent string
	// WaitForCacheSync waits for all the caches to sync. If it exceeds the maximum timeout (seconds), the kubeWatcher start() returns error.
	WaitForCacheSyncTimeout int
}

// WatchOptions is a structure to configure the watching configuration of KubeWatcher.
type WatchOptions struct {
	ControllerName          string
	Predicates              []predicate.Predicate
	EnqueueEventHandler     handler.EventHandler
	MaxConcurrentReconciles int
}

// WatchOption configures how we set up the kube-watch configuration.
// It can guide the kube-watch how to watch the KubeResource
type WatchOption interface {
	Apply(*WatchOptions)
}

// funWatchOption wraps a function that modifies watchOptions into an
// implementation of the WatchOption interface.
type funWatchOption struct {
	f func(*WatchOptions)
}

// Apply applies a watch option
func (fdo *funWatchOption) Apply(do *WatchOptions) {
	fdo.f(do)
}

func newFuncWatchOption(f func(*WatchOptions)) *funWatchOption {
	return &funWatchOption{
		f: f,
	}
}

// WithControllerName creates option with controllerName.
// It configures the controller name.
func WithControllerName(ControllerName string) WatchOption {
	return newFuncWatchOption(func(o *WatchOptions) {
		o.ControllerName = ControllerName
	})
}

// WithPredicates creates option with predicates.
// It configures how to filter events before enqueuing the keys.
func WithPredicates(predicates []predicate.Predicate) WatchOption {
	return newFuncWatchOption(func(o *WatchOptions) {
		o.Predicates = predicates
	})
}

// WithEnqueueEventHandler creates option with eventHandler.
// It configures how to handle the creation, update and deletion of resources into the work queue
func WithEnqueueEventHandler(eventHandler handler.EventHandler) WatchOption {
	return newFuncWatchOption(func(o *WatchOptions) {
		o.EnqueueEventHandler = eventHandler
	})
}

// WithMaxConcurrent creates option with maxConcurrent.
// It configures the maximum number of concurrent Reconciles which can be run.
func WithMaxConcurrent(maxConcurrent int) WatchOption {
	return newFuncWatchOption(func(o *WatchOptions) {
		o.MaxConcurrentReconciles = maxConcurrent
	})
}

// DefaultWatchOptions returns the default ConnOptions.
func DefaultWatchOptions() WatchOptions {
	return WatchOptions{
		Predicates:              []predicate.Predicate{},
		EnqueueEventHandler:     &WatchEnqueueRequestForObject{},
		MaxConcurrentReconciles: 50,
		ControllerName:          "",
	}
}

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
	"fmt"
	"golang.org/x/net/context"
	"k8s.io/client-go/util/workqueue"
	"sigs.k8s.io/controller-runtime/pkg/handler"
	"sigs.k8s.io/controller-runtime/pkg/manager"
	"sigs.k8s.io/controller-runtime/pkg/predicate"
	"sigs.k8s.io/controller-runtime/pkg/ratelimiter"
	"sigs.k8s.io/controller-runtime/pkg/source"
)

// ControllerOptions are the arguments for creating a new Controller
type ControllerOptions struct {
	// MaxConcurrentReconciles is the maximum number of concurrent Reconciles which can be run. Defaults to 50.
	MaxConcurrentReconciles int

	// Reconciler reconciles an object
	Reconciler Reconciler

	// RateLimiter is used to limit how frequently requests may be queued.
	// Defaults to MaxOfRateLimiter which has both overall and per-item rate limiting.
	// The overall is a token bucket and the per-item is exponential.
	RateLimiter ratelimiter.RateLimiter
}

// Controller implements a Kubernetes API.  A Controller manages a work queue fed reconcile.Requests
// from source.Sources.  Work is performed through the reconcile.Reconciler for each enqueued item.
// Work typically is reads and writes Kubernetes objects to make the system state match the state specified
// in the object Spec.
type Controller interface {
	// Reconciler is called to reconcile an object by Namespace/Name
	Reconciler

	// Watch takes events provided by a Source and uses the EventHandler to
	// enqueue reconcile.Requests in response to the events.
	//
	// Watch may be provided one or more Predicates to filter events before
	// they are given to the EventHandler.  Events will be passed to the
	// EventHandler if all provided Predicates evaluate to true.
	Watch(src source.Source, eventhandler handler.EventHandler, predicates ...predicate.Predicate) error

	// Start starts the controller.  Start blocks until stop is closed or a
	// controller has an error starting.
	Start(ctx context.Context) error
}

// NewController returns a new Controller registered with the Manager. The Manager will ensure that shared Caches have
// been synced before the Controller is Started.
func NewController(name string, mgr manager.Manager, options ControllerOptions) (Controller, error) {
	if options.Reconciler == nil {
		return nil, fmt.Errorf("must specify Reconciler")
	}

	if len(name) == 0 {
		return nil, fmt.Errorf("must specify Name for Controller")
	}

	if options.MaxConcurrentReconciles <= 0 {
		options.MaxConcurrentReconciles = 50
	}

	if options.RateLimiter == nil {
		options.RateLimiter = workqueue.DefaultControllerRateLimiter()
	}

	// Inject dependencies into Reconciler
	if err := mgr.SetFields(options.Reconciler); err != nil {
		return nil, err
	}

	// Create controller with dependencies set
	c := &WatchController{
		Do:       options.Reconciler,
		Scheme:   mgr.GetScheme(),
		Client:   mgr.GetClient(),
		Recorder: mgr.GetEventRecorderFor(name),
		MakeQueue: func() workqueue.RateLimitingInterface {
			return workqueue.NewNamedRateLimitingQueue(options.RateLimiter, name)
		},
		MaxConcurrentReconciles: options.MaxConcurrentReconciles,
		Name:                    name,
	}

	// Add the controller as a Manager components
	return c, mgr.Add(c)
}

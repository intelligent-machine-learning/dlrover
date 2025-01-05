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
	"fmt"
	log "github.com/golang/glog"
	"k8s.io/apimachinery/pkg/runtime"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/tools/record"
	"k8s.io/client-go/util/workqueue"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/handler"
	"sigs.k8s.io/controller-runtime/pkg/predicate"
	"sigs.k8s.io/controller-runtime/pkg/runtime/inject"
	"sigs.k8s.io/controller-runtime/pkg/source"
	"sync"
	"time"
)

var _ inject.Injector = &WatchController{}

// Note: This Go file inherits from controller-runtime (open source), the difference here that it add the uuid of kubernetes in WatchRequest
// The uuid of kubernetes is needed actually in cougar monitor. (Using it to reduce the number of rpc access with metadata service)
// https://github.com/kubernetes-sigs/controller-runtime/blob/master/pkg/controller/controller.go

// WatchController implements controller.Controller
type WatchController struct {
	// Name is used to uniquely identify a Controller in tracing, logging and monitoring.  Name is required.
	Name string

	// MaxConcurrentReconciles is the maximum number of concurrent Reconciles which can be run. Defaults to 50.
	MaxConcurrentReconciles int

	// Reconciler is a function that can be called at any time with the Name / Namespace of an object and
	// ensures that the state of the system matches the state specified in the object.
	// Defaults to the DefaultReconcileFunc.
	Do Reconciler

	// Client is a lazily initialized Client.  The controllerManager will initialize this when Start is called.
	Client client.Client

	// Scheme is injected by the controllerManager when controllerManager.Start is called
	Scheme *runtime.Scheme

	// MakeQueue constructs the queue for this controller once the controller is ready to start.
	// This exists because the standard Kubernetes workqueues start themselves immediately, which
	// leads to goroutine leaks if something calls controller.New repeatedly.
	MakeQueue func() workqueue.RateLimitingInterface

	// Queue is an listeningQueue that listens for events from Informers and adds object keys to
	// the Queue for processing
	Queue workqueue.RateLimitingInterface

	// SetFields is used to inject dependencies into other objects such as Sources, EventHandlers and Predicates
	SetFields func(i interface{}) error

	// mu is used to synchronize Controller setup
	mu sync.Mutex

	// JitterPeriod allows tests to reduce the JitterPeriod so they complete faster
	JitterPeriod time.Duration

	// Started is true if the Controller has been Started
	Started bool

	// Recorder is an event recorder for recording Event resources to the
	// Kubernetes API.
	Recorder record.EventRecorder

	// startWatches maintains a list of sources, handlers, and predicates to start when the controller is started.
	startWatches []watchDescription

	stop <-chan struct{}
}

// watchDescription contains all the information necessary to start a watch.
type watchDescription struct {
	src        source.Source
	handler    handler.EventHandler
	predicates []predicate.Predicate
}

// Reconcile implements Reconciler
func (c *WatchController) Reconcile(r WatchRequest) (Result, error) {
	return c.Do.Reconcile(r)
}

// Watch implements Controller
func (c *WatchController) Watch(src source.Source, eventHandler handler.EventHandler, pred ...predicate.Predicate) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	// Inject Cache into arguments
	if err := c.SetFields(src); err != nil {
		return err
	}
	if err := c.SetFields(eventHandler); err != nil {
		return err
	}
	for _, pr := range pred {
		if err := c.SetFields(pr); err != nil {
			return err
		}
	}

	// Controller hasn't started yet, store the watches locally and return.
	//
	// These watches are going to be held on the controller struct until the manager or user calls Start(...).
	if !c.Started {
		c.startWatches = append(c.startWatches, watchDescription{src: src, handler: eventHandler, predicates: pred})
		return nil
	}
	return src.Start(context.Background(), eventHandler, c.Queue, pred...)
}

// Start implements controller.Controller
func (c *WatchController) Start(ctx context.Context) error {
	// use an IIFE to get proper lock handling
	// but lock outside to get proper handling of the queue shutdown
	c.mu.Lock()

	c.Queue = c.MakeQueue()
	defer c.Queue.ShutDown() // needs to be outside the iife so that we shutdown after the stop channel is closed

	err := func() error {
		defer c.mu.Unlock()

		defer utilruntime.HandleCrash()

		// NB(directxman12): launch the sources *before* trying to wait for the
		// caches to sync so that they have a chance to register their intendeded
		// caches.
		for _, watch := range c.startWatches {
			if err := watch.src.Start(ctx, watch.handler, c.Queue, watch.predicates...); err != nil {
				return err
			}
		}

		for _, watch := range c.startWatches {
			syncingSource, ok := watch.src.(source.SyncingSource)
			if !ok {
				continue
			}
			if err := syncingSource.WaitForSync(ctx); err != nil {
				// This code is unreachable in case of kube watches since WaitForCacheSync will never return an error
				// Leaving it here because that could happen in the future
				err = fmt.Errorf("failed to wait for %s caches to sync: %w", c.Name, err)
				log.Errorf("could not wait for Cache to sync: %v", err)
				return err
			}
		}

		// All the watches have been started, we can reset the local slice.
		//
		// We should never hold watches more than necessary, each watch source can hold a backing cache,
		// which won't be garbage collected if we hold a reference to it.
		c.startWatches = nil

		if c.JitterPeriod == 0 {
			c.JitterPeriod = 1 * time.Second
		}

		// Launch workers to process resources
		log.Errorf("starting workers %d", c.MaxConcurrentReconciles)
		for i := 0; i < c.MaxConcurrentReconciles; i++ {
			// Process work items
			go wait.Until(c.worker, c.JitterPeriod, c.stop)
		}

		c.Started = true
		return nil
	}()
	if err != nil {
		return err
	}

	<-c.stop
	log.Info("stopping workers")
	return nil
}

// worker runs a worker thread that just dequeues items, processes them, and marks them done.
// It enforces that the reconcileHandler is never invoked concurrently with the same object.
func (c *WatchController) worker() {
	for c.processNextWorkItem() {
	}
}

// processNextWorkItem will read a single work item off the workqueue and
// attempt to process it, by calling the reconcileHandler.
func (c *WatchController) processNextWorkItem() bool {
	obj, shutdown := c.Queue.Get()
	if shutdown {
		// Stop working
		return false
	}

	// We call Done here so the workqueue knows we have finished
	// processing this item. We also must remember to call Forget if we
	// do not want this work item being re-queued. For example, we do
	// not call Forget if a transient error occurs, instead the item is
	// put back on the workqueue and attempted again after a back-off
	// period.
	defer c.Queue.Done(obj)

	return c.reconcileHandler(obj)
}

func (c *WatchController) reconcileHandler(obj interface{}) bool {
	// Make sure that the the object is a valid request.
	req, ok := obj.(WatchRequest)
	if !ok {
		// As the item in the workqueue is actually invalid, we call
		// Forget here else we'd go into a loop of attempting to
		// process a work item that is invalid.
		c.Queue.Forget(obj)
		log.Errorf("queue item was not a Request: %v", obj)
		// Return true, don't take a break
		return true
	}

	// RunInformersAndControllers the syncHandler, passing it the namespace/Name string of the
	// resource to be synced.
	if result, err := c.Do.Reconcile(req); err != nil {
		c.Queue.AddRateLimited(req)
		log.Errorf("Reconciler error [name:%s] [namespace:%s] with err:%v", req.Name, req.Namespace, err)
		return false
	} else if result.RequeueAfter > 0 {
		// The result.RequeueAfter request will be lost, if it is returned
		// along with a non-nil error. But this is intended as
		// We need to drive to stable reconcile loops before queuing due
		// to result.RequestAfter
		c.Queue.Forget(obj)
		c.Queue.AddAfter(req, result.RequeueAfter)
		return true
	} else if result.Requeue {
		c.Queue.AddRateLimited(req)
		return true
	}

	// Finally, if no error occurs we Forget this item so it does not
	// get queued again until another change happens.
	c.Queue.Forget(obj)

	log.Infof("successfully reconciled [name:%s] [namespace:%s]", req.Name, req.Namespace)

	// Return true, don't take a break
	return true
}

// InjectFunc implement SetFields.Injector
func (c *WatchController) InjectFunc(f inject.Func) error {
	c.SetFields = f
	return nil
}

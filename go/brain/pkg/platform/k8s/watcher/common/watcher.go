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
	"errors"
	"fmt"
	log "github.com/golang/glog"
	"github.com/intelligent-machine-learning/easydl/brain/pkg/common"
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/rest"
	"sigs.k8s.io/controller-runtime/pkg/client/config"
	"sigs.k8s.io/controller-runtime/pkg/manager"
	"sigs.k8s.io/controller-runtime/pkg/scheme"
	"sigs.k8s.io/controller-runtime/pkg/source"
	"time"
)

const (
	// ComponentName is the name of component
	ComponentName = "KubeWatcher"
	// Namespace is the namespace
	Namespace = "namespace"
)

// KubeWatcher is the struct of kube watcher
type KubeWatcher struct {
	controllerManager manager.Manager
	options           *KubeWatchOptions
	ctx               context.Context
	errHandler        common.ErrorHandler
}

// NewKubeWatcher returns a new KubeWatcher
func NewKubeWatcher(ctx context.Context, errHandler common.ErrorHandler, options KubeWatchOptions) (*KubeWatcher, error) {
	kubeWatch := &KubeWatcher{}
	if err := kubeWatch.Initialize(ctx, errHandler, options); err != nil {
		log.Errorf("failed to init the kube-watch with err: %v", err)
		return nil, err
	}
	return kubeWatch, nil
}

// Initialize initializes a KubeWatcher
func (kw *KubeWatcher) Initialize(ctx context.Context, errHandler common.ErrorHandler, options KubeWatchOptions) error {
	log.Info("setting up kubernetes rest cfg for kube-watch")
	kw.options = &options
	kw.errHandler = errHandler

	// Get a cfg to talk to the apiServer
	cfg, err := kw.getConfig()
	if err != nil {
		log.Errorf("failed to new kubernetes rest cfg with err: %v", err)
		return err
	}

	if options.Scheme == nil {
		log.Warning("empty options.scheme. Using default CoreV1Scheme and AppV1Scheme.")
		// if options.scheme = nil , set default value
		newScheme := runtime.NewScheme()
		if err := kw.addCoreV1SchemeToScheme(newScheme); err != nil {
			log.Errorf("failed to add scheme [GroupVersion:%s] with err: %v ", corev1.SchemeGroupVersion.String(), err)
			return err
		}
		if err := kw.addAppV1SchemeToScheme(newScheme); err != nil {
			log.Errorf("failed to add scheme [GroupVersion:%s] with err: %v ", appsv1.SchemeGroupVersion.String(), err)
			return err
		}
		options.Scheme = newScheme
	}

	kw.controllerManager, err = manager.New(cfg, options.Options)
	if err != nil {
		log.Errorf("failed to set up overall controller manager with err: %v", err)
		return err
	}
	kw.ctx = ctx
	return nil
}

// WatchKubeResource watches a particular resources and execute the handler functions to process the events of this resource
func (kw *KubeWatcher) WatchKubeResource(gvk schema.GroupVersionKind, handler EventHandler, watchOpts ...WatchOption) error {
	if gvk.Empty() {
		err := errors.New("invalid schema.GroupVersionKind parameter settings")
		return err
	}
	if handler == nil {
		err := errors.New("null handler parameter settings")
		return err
	}

	controllerManager := kw.controllerManager
	opts := DefaultWatchOptions()
	for _, opt := range watchOpts {
		opt.Apply(&opts)
	}

	obj := &unstructured.Unstructured{}
	obj.SetGroupVersionKind(gvk)

	reconciler := NewWatchReconciler(gvk, controllerManager.GetClient(), controllerManager.GetScheme(), handler)

	kind := &source.Kind{Type: obj}
	// Create a new controller
	controllerName := obj.GetObjectKind().GroupVersionKind().String()
	if opts.ControllerName != "" {
		controllerName = opts.ControllerName
	}
	c, err := NewController(controllerName, controllerManager, ControllerOptions{
		Reconciler:              reconciler,
		MaxConcurrentReconciles: opts.MaxConcurrentReconciles,
	})
	if err != nil {
		return err
	}

	// Watch for changes to resources
	err = c.Watch(kind, opts.EnqueueEventHandler, opts.Predicates...)
	if err != nil {
		return err
	}
	log.Infof("KubeWatch watch the kubernetes resource [GroupVersionKind:%s]", gvk.String())
	return nil
}

// Start starts a KubeWatcher
func (kw *KubeWatcher) Start() error {
	// controllerManger error export by errHandler
	go func(namespace string) {
		if err := kw.controllerManager.Start(kw.ctx); err != nil {
			newError := common.NewError(ComponentName, err)
			newError.ExtraInfos[Namespace] = namespace
			kw.errHandler.ReportError(kw.ctx, newError)
		}
		log.Infof("Exit the controllerManager [namespace:%v] successfully.", namespace)
	}(kw.options.Namespace)

	ctx := kw.ctx
	if kw.options.WaitForCacheSyncTimeout != 0 {
		var cancelFunc context.CancelFunc
		ctx, cancelFunc = context.WithTimeout(kw.ctx, time.Duration(kw.options.WaitForCacheSyncTimeout)*time.Second)
		defer cancelFunc()
	}
	if syncSuccess := kw.controllerManager.GetCache().WaitForCacheSync(ctx); !syncSuccess {
		return fmt.Errorf("sync a cache failed [namespace:%v]", kw.options.Namespace)
	}
	return nil
}

// RegisterKubeResourceScheme registers a resource scheme
func (kw *KubeWatcher) RegisterKubeResourceScheme(schemeBuilder *scheme.Builder) error {
	if schemeBuilder.SchemeBuilder == nil {
		err := errors.New("invalid schemeBuilder parameter settings")
		return err
	}
	if schemeBuilder.GroupVersion.Empty() {
		err := errors.New("invalid schema.GroupVersion parameter settings")
		return err
	}

	if err := schemeBuilder.AddToScheme(kw.controllerManager.GetScheme()); err != nil {
		log.Errorf("failed to add scheme [GroupVersion:%s] with err: %v", schemeBuilder.GroupVersion.String(), err)
		return err
	}
	log.Infof("register kubernetes resource scheme [GroupVersion:%s]", schemeBuilder.GroupVersion.String())
	return nil
}

func (kw *KubeWatcher) getConfig() (*rest.Config, error) {
	cfg, err := config.GetConfig()
	if err != nil {
		return nil, err
	}
	cfg.QPS = kw.options.QPS
	cfg.Burst = kw.options.Burst
	cfg.UserAgent = kw.options.UserAgent
	return cfg, nil
}

func (kw *KubeWatcher) addCoreV1SchemeToScheme(scheme *runtime.Scheme) error {
	if err := corev1.AddToScheme(scheme); err != nil {
		return err
	}
	return nil
}

func (kw *KubeWatcher) addAppV1SchemeToScheme(scheme *runtime.Scheme) error {
	if err := appsv1.AddToScheme(scheme); err != nil {
		return err
	}
	return nil
}

// ControllerManager returns the manager
func (kw *KubeWatcher) ControllerManager() manager.Manager {
	return kw.controllerManager
}

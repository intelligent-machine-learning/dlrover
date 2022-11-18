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

package config

import (
	"context"
	"fmt"
	log "github.com/golang/glog"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/cache"
)

// OnConfigMapChange is the function to be executed when observe the update of a given config map
type OnConfigMapChange func(newCM *corev1.ConfigMap) error

// Controller is the struct of config controller
type Controller struct {
	namespace     string
	name          string
	kubeClientSet kubernetes.Interface
}

// NewController returns a ConfigController watching ConfigMap with specific namespace and name.
func NewController(namespace, name string, kubeClientSet kubernetes.Interface) *Controller {
	return &Controller{
		namespace:     namespace,
		name:          name,
		kubeClientSet: kubeClientSet,
	}
}

// Run start watching ConfigMap and invoke OnConfigMapChange when ConfigMap updates.
func (cc *Controller) Run(ctx context.Context, onChangeFunc OnConfigMapChange) {
	restClient := cc.kubeClientSet.CoreV1().RESTClient()
	resource := "configmaps"
	fieldSelector := fields.ParseSelectorOrDie(fmt.Sprintf("metadata.name=%s", cc.name))
	listFunc := func(options metav1.ListOptions) (runtime.Object, error) {
		options.FieldSelector = fieldSelector.String()
		req := restClient.Get().
			Namespace(cc.namespace).
			Resource(resource).
			VersionedParams(&options, metav1.ParameterCodec)
		return req.Do(ctx).Get()
	}
	watchFunc := func(options metav1.ListOptions) (watch.Interface, error) {
		options.Watch = true
		options.FieldSelector = fieldSelector.String()
		req := restClient.Get().
			Namespace(cc.namespace).
			Resource(resource).
			VersionedParams(&options, metav1.ParameterCodec)
		return req.Watch(ctx)
	}
	source := &cache.ListWatch{ListFunc: listFunc, WatchFunc: watchFunc}
	_, controller := cache.NewInformer(
		source,
		&corev1.ConfigMap{},
		0,
		cache.ResourceEventHandlerFuncs{
			UpdateFunc: func(old, new interface{}) {
				oldCM := old.(*corev1.ConfigMap)
				newCM := new.(*corev1.ConfigMap)
				if oldCM.ResourceVersion == newCM.ResourceVersion {
					return
				}
				log.Infof("Detected ConfigMap update.")
				err := onChangeFunc(newCM)
				if err != nil {
					log.Errorf("Update of config failed due to: %v", err)
				}
			},
		})

	log.Infof("ConfigMap controller start watching: %s", cc.name)
	go controller.Run(ctx.Done())
}

// Get returns the latest ConfigMap in k8s.
func (cc *Controller) Get(ctx context.Context) (*corev1.ConfigMap, error) {
	cmClient := cc.kubeClientSet.CoreV1().ConfigMaps(cc.namespace)
	cm, err := cmClient.Get(ctx, cc.name, metav1.GetOptions{})
	if err != nil {
		log.Errorf("Get ConfigMap error: %v", err)
		return nil, err
	}
	return cm, nil
}

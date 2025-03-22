/*
Copyright 2022.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package main

import (
	"flag"
	"fmt"
	"os"

	// Import all Kubernetes client auth plugins (e.g. Azure, GCP, OIDC, etc.)
	// to ensure that exec-entrypoint and run can make use of them.
	_ "k8s.io/client-go/plugin/pkg/client/auth"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	clientgoscheme "k8s.io/client-go/kubernetes/scheme"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/cache"
	"sigs.k8s.io/controller-runtime/pkg/controller"
	"sigs.k8s.io/controller-runtime/pkg/event"
	"sigs.k8s.io/controller-runtime/pkg/handler"
	"sigs.k8s.io/controller-runtime/pkg/healthz"
	"sigs.k8s.io/controller-runtime/pkg/log/zap"
	"sigs.k8s.io/controller-runtime/pkg/predicate"
	"sigs.k8s.io/controller-runtime/pkg/source"

	"github.com/intelligent-machine-learning/dlrover/go/elasticjob/api/v1alpha1"
	elasticv1alpha1 "github.com/intelligent-machine-learning/dlrover/go/elasticjob/api/v1alpha1"
	"github.com/intelligent-machine-learning/dlrover/go/elasticjob/pkg/common"
	"github.com/intelligent-machine-learning/dlrover/go/elasticjob/pkg/controllers"
	"github.com/sirupsen/logrus"
	//+kubebuilder:scaffold:imports
)

var (
	scheme   = runtime.NewScheme()
	setupLog = ctrl.Log.WithName("main")
)

func init() {
	utilruntime.Must(clientgoscheme.AddToScheme(scheme))

	utilruntime.Must(elasticv1alpha1.AddToScheme(scheme))
	//+kubebuilder:scaffold:scheme
}

func main() {
	var metricsAddr string
	var enableLeaderElection bool
	var probeAddr string
	var masterImage string
	flag.StringVar(&metricsAddr, "metrics-bind-address", ":8080", "The address the metric endpoint binds to.")
	flag.StringVar(&probeAddr, "health-probe-bind-address", ":8081", "The address the probe endpoint binds to.")
	flag.BoolVar(&enableLeaderElection, "leader-elect", false,
		"Enable leader election for controller manager. "+
			"Enabling this will ensure there is only one active controller manager.")
	flag.StringVar(&masterImage, "master-image", "registry.cn-hangzhou.aliyuncs.com/intell-ai/dlrover:latest",
		"The image to launch a dlrover master Pod of an ElasticJob.")
	opts := zap.Options{
		Development: true,
	}
	opts.BindFlags(flag.CommandLine)
	flag.Parse()

	logrus.Infof("The default master image is %s", masterImage)

	ctrl.SetLogger(zap.New(zap.UseFlagOptions(&opts)))

	mgr, err := ctrl.NewManager(ctrl.GetConfigOrDie(), ctrl.Options{
		Scheme:                 scheme,
		MetricsBindAddress:     metricsAddr,
		Port:                   9443,
		HealthProbeBindAddress: probeAddr,
		LeaderElection:         enableLeaderElection,
		LeaderElectionID:       "dlrover-elasticjob",
		NewCache:               createCacheFunc(),
	})
	if err != nil {
		setupLog.Error(err, "unable to start manager")
		os.Exit(1)
	}

	elasticJobReconciler := controllers.NewElasticJobReconciler(mgr, masterImage)
	c, err := controller.New("elasticjob", mgr, controller.Options{
		Reconciler: elasticJobReconciler,
	})
	if err != nil {
		setupLog.Error(err, "unable to create controller", "controller", "ElasticJob")
		os.Exit(1)
	}

	err = c.Watch(&source.Kind{Type: &v1alpha1.ElasticJob{}}, &handler.EnqueueRequestForObject{}, predicate.Funcs{
		CreateFunc: func(createEvent event.CreateEvent) bool {
			return true
		},
		UpdateFunc: func(updateEvent event.UpdateEvent) bool {
			return updateEvent.ObjectOld.GetResourceVersion() != updateEvent.ObjectNew.GetResourceVersion()
		},
		DeleteFunc: func(deleteEvent event.DeleteEvent) bool {
			job := deleteEvent.Object.(*v1alpha1.ElasticJob)
			delete(elasticJobReconciler.CachedJobs, job.Name)
			setupLog.Info(fmt.Sprintf("Remove the job %s from cached buffer.", job.Name))
			return false
		},
	})
	if err != nil {
		setupLog.Error(err, "fail to set watching ElasticJobs.")
	}

	enqueue := &controllers.EnqueueRequestForPod{
		Reconciler: elasticJobReconciler,
	}
	err = c.Watch(&source.Kind{Type: &corev1.Pod{}}, enqueue, predicate.Funcs{
		CreateFunc: func(createEvent event.CreateEvent) bool {
			return true
		},
		UpdateFunc: func(updateEvent event.UpdateEvent) bool {
			oldPod := updateEvent.ObjectOld.(*corev1.Pod)
			newPod := updateEvent.ObjectNew.(*corev1.Pod)
			if oldPod.ResourceVersion == newPod.ResourceVersion ||
				oldPod.Status.Phase == corev1.PodFailed ||
				oldPod.Status.Phase == corev1.PodSucceeded {
				return false
			}
			return true
		},
		DeleteFunc: func(deleteEvent event.DeleteEvent) bool {
			return true
		},
	})
	if err != nil {
		setupLog.Error(err, "fail to set watching Pods.")
	}

	//+kubebuilder:scaffold:builder
	if err := mgr.AddHealthzCheck("healthz", healthz.Ping); err != nil {
		setupLog.Error(err, "unable to set up health check")
		os.Exit(1)
	}
	if err := mgr.AddReadyzCheck("readyz", healthz.Ping); err != nil {
		setupLog.Error(err, "unable to set up ready check")
		os.Exit(1)
	}

	setupLog.Info("starting manager")
	if err := mgr.Start(ctrl.SetupSignalHandler()); err != nil {
		setupLog.Error(err, "problem running manager")
		os.Exit(1)
	}
}

func createCacheFunc() cache.NewCacheFunc {
	// Only watch the dlrover master Pod to recover the Pod or decide the job status.
	// The dlrover master will watch the other Pods and decide the scale plan to
	// notify the elasticjob controller to modify Pods.
	return cache.BuilderWithOptions(cache.Options{
		Scheme: scheme,
		SelectorsByObject: cache.SelectorsByObject{
			&corev1.Pod{}: {
				Label: labels.SelectorFromSet(labels.Set{
					common.LabelAppNameKey: common.ElasticJobAppName,
				}),
			},
		},
	})
}

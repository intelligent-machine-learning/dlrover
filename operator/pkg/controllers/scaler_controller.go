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

package controllers

import (
	"context"
	"time"

	"github.com/go-logr/logr"
	logger "github.com/sirupsen/logrus"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/tools/record"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"

	elasticv1alpha1 "github.com/intelligent-machine-learning/easydl/operator/api/v1alpha1"
)

const (
	pollInterval = time.Duration(3 * time.Second)
)

// ScaleReconciler reconciles an ElasticJob object
type ScalerReconciler struct {
	client.Client
	Scheme   *runtime.Scheme
	Recorder record.EventRecorder
	Log      logr.Logger
}

//+kubebuilder:rbac:groups=elastic.iml.github.io,resources=scaler,verbs=get;list;watch;create;update;patch;delete
//+kubebuilder:rbac:groups=elastic.iml.github.io,resources=scaler/status,verbs=get;update;patch
//+kubebuilder:rbac:groups=elastic.iml.github.io,resources=scaler/finalizers,verbs=update

// Reconcile is part of the main kubernetes reconciliation loop which aims to
// move the current state of the cluster closer to the desired state.
// TODO(user): Modify the Reconcile function to compare the state specified by
// the ElasticJob object against the actual cluster state, and then
// perform operations to make the cluster state reflect the state specified by
// the user.
//
// For more details, check Reconcile and its Result here:
// - https://pkg.go.dev/sigs.k8s.io/controller-runtime@v0.12.2/pkg/reconcile
func (r *ScalerReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	_ = log.FromContext(ctx)

	// rlog := r.Log.WithValues("scaler", req.NamespacedName)
	// Fetch the scale
	scaler := &elasticv1alpha1.Scaler{}
	if err := r.Get(context.TODO(), req.NamespacedName, scaler); err != nil {
		if errors.IsNotFound(err) {
			return ctrl.Result{}, nil
		}
		return ctrl.Result{}, err
	}

	return r.setScalingOwner(scaler, pollInterval)
}

func (r *ScalerReconciler) getOwnerJob(scaler *elasticv1alpha1.Scaler) *elasticv1alpha1.ElasticJob {
	job := &elasticv1alpha1.ElasticJob{}
	nsn := types.NamespacedName{}
	nsn.Namespace = scaler.GetNamespace()
	nsn.Name = scaler.Spec.OwnerJob
	err := r.Get(context.Background(), nsn, job)
	if err != nil {
		if errors.IsNotFound(err) {
			logger.Warnf("%s not found elasticJob: %v, namespace: %v", scaler.Name, nsn.Name, nsn.Namespace)
			return nil
		}
		return nil
	}
	return job
}

func (r *ScalerReconciler) setScalingOwner(scaler *elasticv1alpha1.Scaler, pollInterval time.Duration) (ctrl.Result, error) {
	ownerRefs := scaler.GetOwnerReferences()
	if len(ownerRefs) == 0 {
		job := r.getOwnerJob(scaler)
		gvk := elasticv1alpha1.SchemeGroupVersionKind
		ownerRefs = append(ownerRefs, *metav1.NewControllerRef(job, schema.GroupVersionKind{Group: gvk.Group, Version: gvk.Version, Kind: gvk.Kind}))
		scaler.SetOwnerReferences(ownerRefs)

		err := r.Status().Update(context.Background(), scaler)
		if err != nil {
			logger.Errorf("failed to update scaler status: %s, err: %++v", scaler.Name, err)
			return ctrl.Result{RequeueAfter: pollInterval}, nil
		}

		err = r.Update(context.Background(), scaler)
		if err != nil {
			logger.Errorf("failed to update scaler: %s, err: %++v", scaler.Name, err)
			// Error updating the scaler - requeue the request.
			return ctrl.Result{RequeueAfter: pollInterval}, nil
		}
	}
	return ctrl.Result{}, nil
}

// SetupWithManager sets up the controller with the Manager.
func (r *ScalerReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&elasticv1alpha1.Scaler{}).
		Complete(r)
}

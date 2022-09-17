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
	"fmt"

	"github.com/go-logr/logr"
	logger "github.com/sirupsen/logrus"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/tools/record"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"

	elasticv1alpha1 "github.com/intelligent-machine-learning/easydl/operator/api/v1alpha1"
	commonv1 "github.com/kubeflow/common/pkg/apis/common/v1"
	commonutil "github.com/kubeflow/common/pkg/util"
)

// ElasticJobReconciler reconciles a ElasticJob object
type ElasticJobReconciler struct {
	client.Client
	Scheme   *runtime.Scheme
	Recorder record.EventRecorder
	Log      logr.Logger
}

var masterManager *MasterManager

func init() {
	masterManager = newMasterManager()
}

//+kubebuilder:rbac:groups=elastic.iml.github.io,resources=elasticjobs,verbs=get;list;watch;create;update;patch;delete
//+kubebuilder:rbac:groups=elastic.iml.github.io,resources=elasticjobs/status,verbs=get;update;patch
//+kubebuilder:rbac:groups=elastic.iml.github.io,resources=elasticjobs/finalizers,verbs=update

// Reconcile is part of the main kubernetes reconciliation loop which aims to
// move the current state of the cluster closer to the desired state.
// TODO(user): Modify the Reconcile function to compare the state specified by
// the ElasticJob object against the actual cluster state, and then
// perform operations to make the cluster state reflect the state specified by
// the user.
//
// For more details, check Reconcile and its Result here:
// - https://pkg.go.dev/sigs.k8s.io/controller-runtime@v0.12.2/pkg/reconcile
func (r *ElasticJobReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	_ = log.FromContext(ctx)

	rlog := r.Log.WithValues("elasticjob", req.NamespacedName)
	// Fetch the elastic Training job
	job := &elasticv1alpha1.ElasticJob{}
	if err := r.Get(context.TODO(), req.NamespacedName, job); err != nil {
		if errors.IsNotFound(err) {
			// Object not found, return.  Created objects are automatically garbage collected.
			// For additional cleanup logic use finalizers.
			return ctrl.Result{}, nil
		}
		// Error reading the object - requeue the request.
		return ctrl.Result{}, err
	}

	if job.DeletionTimestamp != nil {
		rlog.Info("Reconcil cancelled, the job has been deleted")
		return ctrl.Result{}, nil
	}

	r.Scheme.Default(job)

	return r.reconcileJobs(job)
}

func (r *ElasticJobReconciler) reconcileJobs(job *elasticv1alpha1.ElasticJob) (ctrl.Result, error) {
	logger.Infof("jobName: %s, phase %s", job.Name, job.Status.Phase)

	defer func() {
		latestJob := &elasticv1alpha1.ElasticJob{}
		err := r.Get(context.TODO(), types.NamespacedName{
			Name:      job.Name,
			Namespace: job.Namespace,
		}, latestJob)
		if err == nil {
			if latestJob.ObjectMeta.ResourceVersion != job.ObjectMeta.ResourceVersion {
				latestJob.Status = job.Status
				job = latestJob
			}
		}
		err = r.Status().Update(context.TODO(), job)
		if err != nil {
			logger.Warningf("Failed to update %s : %s, error: %v",
				job.GetObjectKind().GroupVersionKind(),
				job.GetName(), err)
		}
	}()

	switch job.Status.Phase {
	case "", commonv1.JobCreated:
		r.initializeJob(job)
		err := r.createEasydlMaster(job)
		if err != nil {
			logger.Warningf("Fail to create EasyDL Master")
		}
		msg := fmt.Sprintf("ElasticJob %s is running.", job.Name)
		updateStatus(&job.Status, commonv1.JobRunning, commonutil.JobRunningReason, msg)
	case commonv1.JobRunning:
		masterManager.syncMasterState(r, job)
	case commonv1.JobSucceeded:
		logger.Infof("Job %s succeed", job.Name)
	default:
		logger.Warningf("job %s unknown status %s", job.Name, job.Status.Phase)
	}
	return ctrl.Result{}, nil
}

func (r *ElasticJobReconciler) initializeJob(job *elasticv1alpha1.ElasticJob) {
	if job.Status.Conditions == nil {
		initializeJobStatuses(&job.Status, ReplicaTypeEasydlMaster)
		msg := fmt.Sprintf("ElasticJob %s is created.", job.Name)
		updateStatus(&job.Status, commonv1.JobCreated, commonutil.JobCreatedReason, msg)
	}
	if job.Status.StartTime == nil {
		now := metav1.Now()
		job.Status.StartTime = &now
	}
}

func (r *ElasticJobReconciler) createEasydlMaster(job *elasticv1alpha1.ElasticJob) error {
	err := masterManager.createPod(r, job)
	if err != nil {
		return err
	}
	return err
}

// SetupWithManager sets up the controller with the Manager.
func (r *ElasticJobReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&elasticv1alpha1.ElasticJob{}).
		Owns(&corev1.Pod{}).
		Complete(r)
}

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
	"time"

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

	elasticv1alpha1 "github.com/intelligent-machine-learning/easydl/dlrover/go/operator/api/v1alpha1"
	common "github.com/intelligent-machine-learning/easydl/dlrover/go/operator/pkg/common"
	commonv1 "github.com/intelligent-machine-learning/easydl/dlrover/go/operator/pkg/common/api/v1"
	master "github.com/intelligent-machine-learning/easydl/dlrover/go/operator/pkg/controllers/master"
)

const (
	defaultPollInterval = time.Duration(5 * time.Second)
)

// ElasticJobReconciler reconciles a ElasticJob object
type ElasticJobReconciler struct {
	client.Client
	Scheme      *runtime.Scheme
	Recorder    record.EventRecorder
	Log         logr.Logger
	masterImage string
}

// NewElasticJobReconciler creates a JobReconciler
func NewElasticJobReconciler(mgr ctrl.Manager, masterImage string) *ElasticJobReconciler {
	r := &ElasticJobReconciler{
		Client:      mgr.GetClient(),
		Scheme:      mgr.GetScheme(),
		Recorder:    mgr.GetEventRecorderFor("elasticjob-controller"),
		Log:         ctrl.Log.WithName("controllers").WithName("ElasticJob"),
		masterImage: masterImage,
	}
	return r
}

//+kubebuilder:rbac:groups=elastic.iml.github.io,resources=elasticjobs,verbs=get;list;watch;create;update;patch;delete
//+kubebuilder:rbac:groups=elastic.iml.github.io,resources=elasticjobs/status,verbs=get;update;patch
//+kubebuilder:rbac:groups=elastic.iml.github.io,resources=elasticjobs/finalizers,verbs=update
//+kubebuilder:rbac:groups=elastic.iml.github.io,resources=scaleplans,verbs=get;list;watch;create;update;patch;delete
//+kubebuilder:rbac:groups=elastic.iml.github.io,resources=scaleplans/status,verbs=get;update;patch
//+kubebuilder:rbac:groups=elastic.iml.github.io,resources=scaleplans/finalizers,verbs=update
//+kubebuilder:rbac:groups="",resources=pods,verbs=create;get;watch;list;update;patch;delete
//+kubebuilder:rbac:groups="",resources=services,verbs=create;get;watch;list;update;patch;delete

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

	defer func() { updateElasticJobStatus(r.Client, job) }()

	switch job.Status.Phase {
	case "", commonv1.JobCreated:
		r.initializeJob(job)
		err := r.createEasydlMaster(job)
		if err != nil {
			logger.Warningf("Fail to create EasyDL Master")
			return ctrl.Result{RequeueAfter: defaultPollInterval}, err
		}
		r.syncJobStateByReplicas(job)
	case commonv1.JobPending:
		r.syncJobStateByReplicas(job)
	case commonv1.JobRunning:
		r.handleFaultPods(job)
		r.syncJobStateByReplicas(job)
	case commonv1.JobScaling:
		scalePlan, err := r.getJobScalePlan(job)
		if err != nil {
			logger.Errorf("Job %s: Fail to get scaleplan: %s", job.Name, err)
		}
		if scalePlan.Status.Phase != commonv1.JobPending {
			logger.Infof("Job %s: Skip a %s scaleplan %s.", job.Name, scalePlan.Status.Phase, scalePlan.Name)
			return ctrl.Result{}, nil
		}
		r.updateScalePlanScaling(scalePlan)
		if scalePlan != nil {
			err := r.executeScaling(job, scalePlan)
			if err != nil {
				logger.Errorf("Job %s: Fail to execute scaleplan %s: %s", job.Name, scalePlan.Name, err)
			}
		}
		r.syncJobStateByReplicas(job)
	case commonv1.JobSucceeded:
		logger.Infof("Job %s succeed", job.Name)
		r.syncJobStateByReplicas(job)
		r.stopRunningPods(job)
	case commonv1.JobFailed:
		logger.Infof("Job %s failed", job.Name)
		r.syncJobStateByReplicas(job)
		r.stopRunningPods(job)
	default:
		logger.Warningf("job %s unknown status %s", job.Name, job.Status.Phase)
	}
	return ctrl.Result{}, nil
}

func (r *ElasticJobReconciler) initializeJob(job *elasticv1alpha1.ElasticJob) {
	if job.Status.Conditions == nil {
		common.InitializeJobStatuses(&job.Status, master.ReplicaTypeJobMaster)
		msg := fmt.Sprintf("ElasticJob %s is created.", job.Name)
		common.UpdateStatus(&job.Status, commonv1.JobCreated, common.JobCreatedReason, msg)
	}
	if job.Status.StartTime == nil {
		now := metav1.Now()
		job.Status.StartTime = &now
	}
}

func (r *ElasticJobReconciler) syncJobStateByReplicas(job *elasticv1alpha1.ElasticJob) {
	for _, manager := range common.ReplicaManagers {
		manager.SyncJobState(r.Client, job)
	}
}

func (r *ElasticJobReconciler) stopRunningPods(job *elasticv1alpha1.ElasticJob) {
	for _, manager := range common.ReplicaManagers {
		manager.StopRunningPods(r.Client, job)
	}
}

func (r *ElasticJobReconciler) createEasydlMaster(job *elasticv1alpha1.ElasticJob) error {
	master.NewMasterTemplateToJob(job, r.masterImage)
	masterManager := common.ReplicaManagers[master.ReplicaTypeJobMaster]
	err := masterManager.ReconcilePods(r.Client, job, nil)
	if err != nil {
		r.Recorder.Eventf(
			job,
			corev1.EventTypeWarning,
			string(commonv1.JobFailed),
			"master pod created failed: %v",
			err,
		)
		return err
	}
	return err
}

func (r *ElasticJobReconciler) getJobScalePlan(job *elasticv1alpha1.ElasticJob) (*elasticv1alpha1.ScalePlan, error) {
	scalePlan := &elasticv1alpha1.ScalePlan{}
	nsn := types.NamespacedName{}
	nsn.Namespace = job.GetNamespace()
	nsn.Name = job.Status.ScalePlan
	err := r.Get(context.Background(), nsn, scalePlan)
	if err != nil {
		if errors.IsNotFound(err) {
			logger.Warnf("ScalePlan %s not found: namespace: %v", nsn.Name, nsn.Namespace)
			return nil, nil
		}
		return nil, err
	}
	return scalePlan, err
}

func (r *ElasticJobReconciler) executeScaling(job *elasticv1alpha1.ElasticJob, scalePlan *elasticv1alpha1.ScalePlan) error {
	for replicaType, replicaManager := range common.ReplicaManagers {
		if replicaType == master.ReplicaTypeJobMaster {
			continue
		}
		err := replicaManager.ReconcilePods(r.Client, job, scalePlan)
		if err != nil {
			r.Recorder.Eventf(
				job,
				corev1.EventTypeWarning,
				"Failed",
				"Reconcile replica %s created failed: %v",
				replicaType,
				err,
			)
			return err
		}
	}
	err := r.updateScalePlanSucceeded(scalePlan)
	return err
}

func (r *ElasticJobReconciler) updateScalePlanSucceeded(scalePlan *elasticv1alpha1.ScalePlan) error {
	now := metav1.Now()
	scalePlan.Status.Phase = commonv1.JobSucceeded
	scalePlan.Status.FinishTime = &now
	err := updateScalePlanStatus(r.Client, scalePlan)
	return err
}

func (r *ElasticJobReconciler) updateScalePlanScaling(scalePlan *elasticv1alpha1.ScalePlan) error {
	scalePlan.Status.Phase = commonv1.JobScaling
	err := updateScalePlanStatus(r.Client, scalePlan)
	return err
}

func (r *ElasticJobReconciler) handleFaultPods(job *elasticv1alpha1.ElasticJob) {
	for replicaType, manager := range common.ReplicaManagers {
		if replicaType == master.ReplicaTypeJobMaster {
			master.NewMasterTemplateToJob(job, r.masterImage)
		}
		manager.HandleFaultPods(r.Client, job)
	}
}

// SetupWithManager sets up the controller with the Manager.
func (r *ElasticJobReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&elasticv1alpha1.ElasticJob{}).
		Owns(&corev1.Pod{}).
		Complete(r)
}

func updateElasticJobStatus(client client.Client, job *elasticv1alpha1.ElasticJob) error {
	latestJob := &elasticv1alpha1.ElasticJob{}
	err := client.Get(context.TODO(), types.NamespacedName{
		Name:      job.Name,
		Namespace: job.Namespace,
	}, latestJob)
	if err == nil {
		if latestJob.ObjectMeta.ResourceVersion != job.ObjectMeta.ResourceVersion {
			latestJob.Status = job.Status
			job = latestJob
		}
	}
	err = client.Status().Update(context.TODO(), job)
	if err != nil {
		logger.Warningf("Failed to update %s : %s, error: %v",
			job.GetObjectKind().GroupVersionKind(),
			job.GetName(), err)
	}
	return err
}

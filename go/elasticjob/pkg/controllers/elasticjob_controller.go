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
	"sigs.k8s.io/controller-runtime/pkg/event"
	"sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/intelligent-machine-learning/dlrover/go/elasticjob/api/v1alpha1"
	elasticv1alpha1 "github.com/intelligent-machine-learning/dlrover/go/elasticjob/api/v1alpha1"
	common "github.com/intelligent-machine-learning/dlrover/go/elasticjob/pkg/common"
	apiv1 "github.com/intelligent-machine-learning/dlrover/go/elasticjob/pkg/common/api/v1"
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
	CachedJobs  map[string]*elasticv1alpha1.ElasticJob
	masterImage string
}

// NewElasticJobReconciler creates a JobReconciler
func NewElasticJobReconciler(mgr ctrl.Manager, masterImage string) *ElasticJobReconciler {
	r := &ElasticJobReconciler{
		Client:      mgr.GetClient(),
		Scheme:      mgr.GetScheme(),
		Recorder:    mgr.GetEventRecorderFor("elasticjob-controller"),
		Log:         ctrl.Log.WithName("controllers").WithName("ElasticJob"),
		CachedJobs:  make(map[string]*elasticv1alpha1.ElasticJob),
		masterImage: masterImage,
	}
	return r
}

//+kubebuilder:rbac:groups=elastic.iml.github.io,resources=elasticjobs,verbs=get;list;watch;create;update;patch;delete
//+kubebuilder:rbac:groups=elastic.iml.github.io,resources=elasticjobs/status,verbs=get;update;patch
//+kubebuilder:rbac:groups=elastic.iml.github.io,resources=elasticjobs/finalizers,verbs=update
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
	if _, ok := r.CachedJobs[req.Name]; !ok {
		// Fetch the elastic Training job
		job, err := r.fetchElasticJob(req.NamespacedName)
		if job == nil {
			if errors.IsNotFound(err) {
				return ctrl.Result{}, nil
			}
			return ctrl.Result{}, err
		}
		if job.DeletionTimestamp != nil {
			rlog.Info("Reconcile cancelled, the job has been deleted")
			return ctrl.Result{}, nil
		}
		r.Scheme.Default(job)
		r.CachedJobs[req.Name] = job
	}
	job := r.CachedJobs[req.Name]
	return r.reconcileJobs(job)
}

func (r *ElasticJobReconciler) fetchElasticJob(
	namespacedName types.NamespacedName) (*elasticv1alpha1.ElasticJob, error) {
	job := &elasticv1alpha1.ElasticJob{}
	if err := r.Get(context.Background(), namespacedName, job); err != nil {
		return nil, err
	}
	return job, nil
}

func (r *ElasticJobReconciler) reconcileJobs(job *elasticv1alpha1.ElasticJob) (ctrl.Result, error) {
	logger.Infof("jobName: %s, phase %s", job.Name, job.Status.Phase)

	defer func() { updateElasticJobStatus(r.Client, job) }()

	now := metav1.Now()
	job.Status.LastReconcileTime = &now

	switch job.Status.Phase {
	case "":
		// A job yaml is applied.
		err := r.createEasticJobMaster(job)
		if err != nil {
			logger.Warningf("Fail to create the elastic job master")
			return ctrl.Result{RequeueAfter: defaultPollInterval}, err
		}
		common.InitializeJobStatuses(&job.Status, JobMasterReplicaType)
		msg := fmt.Sprintf("ElasticJob %s is created.", job.Name)
		common.UpdateJobStatus(&job.Status, apiv1.JobCreated, common.JobCreatedReason, msg)
		if job.Status.StartTime == nil {
			now := metav1.Now()
			job.Status.StartTime = &now
		}
	case apiv1.JobCreated:
		if _, ok := job.Status.ReplicaStatuses[JobMasterReplicaType]; !ok {
			// The job master is not created now.
			return ctrl.Result{}, nil
		}
	case apiv1.JobPending, apiv1.JobRunning:
		masterStatus := job.Status.ReplicaStatuses[JobMasterReplicaType]
		if masterStatus.Initial+masterStatus.Pending+masterStatus.Active == 0 {
			err := r.createEasticJobMaster(job)
			if err != nil {
				logger.Warningf("Fail to create the master of elasticjob %s", job.Name)
				return ctrl.Result{RequeueAfter: defaultPollInterval}, err
			}
		}
	case apiv1.JobSucceeded:
		logger.Infof("Job %s succeed", job.Name)
		delete(r.CachedJobs, job.Name)
	case apiv1.JobFailed:
		logger.Infof("Job %s failed", job.Name)
		delete(r.CachedJobs, job.Name)
	default:
		logger.Warningf("job %s unknown status %s", job.Name, job.Status.Phase)
	}
	return ctrl.Result{}, nil
}

func (r *ElasticJobReconciler) createEasticJobMaster(job *elasticv1alpha1.ElasticJob) error {
	SetDefaultMasterTemplateToJob(job, r.masterImage)
	masterIndex := int32(0)
	if status, ok := job.Status.ReplicaStatuses[JobMasterReplicaType]; ok {
		masterIndex = status.Initial + status.Pending + status.Active + status.Failed
	}

	err := ReconcileJobMasterPod(r.Client, job, masterIndex)
	if err != nil {
		r.Recorder.Eventf(
			job,
			corev1.EventTypeWarning,
			string(apiv1.JobFailed),
			"master pod created failed: %v",
			err,
		)
		return err
	}
	return err
}

func (r *ElasticJobReconciler) ProcessPodCreateEvent(createEvent *event.CreateEvent) bool {
	pod := createEvent.Object.(*corev1.Pod)
	ownerJob := r.getPodOwnerElasticJob(pod)
	if ownerJob == nil {
		return false
	}
	incrementReplicaStatus(pod, ownerJob)
	return true
}

func (r *ElasticJobReconciler) ProcessPodUpdateEvent(updateEvent *event.UpdateEvent) bool {
	oldPod := updateEvent.ObjectOld.(*corev1.Pod)
	newPod := updateEvent.ObjectNew.(*corev1.Pod)
	if oldPod.ResourceVersion == newPod.ResourceVersion ||
		oldPod.Status.Phase == corev1.PodFailed ||
		oldPod.Status.Phase == corev1.PodSucceeded {
		return false
	}
	ownerJob := r.getPodOwnerElasticJob(newPod)
	if ownerJob == nil {
		return false
	}
	incrementReplicaStatus(newPod, ownerJob)
	decreaseReplicaStatus(oldPod, ownerJob)
	if newPod.Labels[common.LabelReplicaTypeKey] == string(JobMasterReplicaType) {
		updateJobStatusPhase(newPod, ownerJob)
	}
	return true
}

func (r *ElasticJobReconciler) ProcessPodDeleteEvent(deleteEvent *event.DeleteEvent) bool {
	pod := deleteEvent.Object.(*corev1.Pod)
	ownerJob := r.getPodOwnerElasticJob(pod)
	if ownerJob == nil {
		return false
	}
	if replicaTypeStr, ok := pod.Labels[common.LabelReplicaTypeKey]; ok {
		replicaType := apiv1.ReplicaType(replicaTypeStr)
		if _, ok := ownerJob.Status.JobStatus.ReplicaStatuses[replicaType]; !ok {
			ownerJob.Status.JobStatus.ReplicaStatuses[replicaType] = &apiv1.ReplicaStatus{}
		}
		replicaStatus := ownerJob.Status.JobStatus.ReplicaStatuses[replicaType]
		if pod.Status.Phase == corev1.PodPending {
			replicaStatus.Pending -= 1
			replicaStatus.Failed += 1
		} else if pod.Status.Phase == corev1.PodRunning {
			replicaStatus.Active -= 1
			replicaStatus.Failed += 1
		}
	}
	return true
}

func incrementReplicaStatus(pod *corev1.Pod, ownerJob *v1alpha1.ElasticJob) {
	if replicaTypeStr, ok := pod.Labels[common.LabelReplicaTypeKey]; ok {
		replicaType := apiv1.ReplicaType(replicaTypeStr)
		if _, ok := ownerJob.Status.JobStatus.ReplicaStatuses[replicaType]; !ok {
			ownerJob.Status.JobStatus.ReplicaStatuses[replicaType] = &apiv1.ReplicaStatus{}
		}
		replicaStatus := ownerJob.Status.JobStatus.ReplicaStatuses[replicaType]
		if pod.Status.Phase == corev1.PodPending {
			replicaStatus.Pending += 1
		} else if pod.Status.Phase == corev1.PodRunning {
			replicaStatus.Active += 1
		} else if pod.Status.Phase == corev1.PodFailed {
			replicaStatus.Failed += 1
		} else if pod.Status.Phase == corev1.PodSucceeded {
			replicaStatus.Succeeded += 1
		}
	}
}

func decreaseReplicaStatus(pod *corev1.Pod, ownerJob *v1alpha1.ElasticJob) {
	if replicaTypeStr, ok := pod.Labels[common.LabelReplicaTypeKey]; ok {
		replicaType := apiv1.ReplicaType(replicaTypeStr)
		if _, ok := ownerJob.Status.JobStatus.ReplicaStatuses[replicaType]; !ok {
			ownerJob.Status.JobStatus.ReplicaStatuses[replicaType] = &apiv1.ReplicaStatus{}
		}
		replicaStatus := ownerJob.Status.JobStatus.ReplicaStatuses[replicaType]
		if pod.Status.Phase == corev1.PodPending {
			replicaStatus.Pending -= 1
		} else if pod.Status.Phase == corev1.PodRunning {
			replicaStatus.Active -= 1
		} else if pod.Status.Phase == corev1.PodFailed {
			replicaStatus.Failed -= 1
		} else if pod.Status.Phase == corev1.PodSucceeded {
			replicaStatus.Succeeded -= 1
		}
	}
}

func updateJobStatusPhase(masterPod *corev1.Pod, job *v1alpha1.ElasticJob) {
	if masterPod.Status.Phase == corev1.PodPending && len(job.Status.ReplicaStatuses) == 1 {
		// The job status is pending if there is only the pending master pod.
		job.Status.Phase = apiv1.JobPending
	} else if masterPod.Status.Phase == corev1.PodFailed {
		exitCode := masterPod.Status.ContainerStatuses[0].State.Terminated.ExitCode
		failedCount := job.Status.ReplicaStatuses[JobMasterReplicaType].Failed
		restartCount := int32(job.Spec.ReplicaSpecs[JobMasterReplicaType].RestartCount)
		if exitCode == 1 {
			// Set the job failed if the master failed with exitcode=1 or the failed count is bigger than
			// the restart count. The exitcode=1 means that the job master raised an exception.
			msg := fmt.Sprintf("The job master %s failed with exitcode=1.", masterPod.Name)
			common.UpdateJobStatus(&job.Status, apiv1.JobFailed, common.JobFailedReason, msg)
		} else if failedCount > restartCount {
			msg := fmt.Sprintf("The job master failover count %d is beyond the restart count %d.",
				failedCount, restartCount)
			common.UpdateJobStatus(&job.Status, apiv1.JobFailed, common.JobFailedReason, msg)
		} else {
			// The controller will relaunch the master pod, so the job status is still running.
			msg := "The job master is failover."
			common.UpdateJobStatus(&job.Status, apiv1.JobRunning, common.JobRestartingReason, msg)
		}
	} else if masterPod.Status.Phase == corev1.PodSucceeded {
		msg := "The job master is succeeded."
		common.UpdateJobStatus(&job.Status, apiv1.JobSucceeded, common.JobSucceededReason, msg)
	} else {
		msg := "The job master is running."
		common.UpdateJobStatus(&job.Status, apiv1.JobRunning, common.JobRunningReason, msg)
	}
}

func (r *ElasticJobReconciler) getPodOwnerElasticJob(pod *corev1.Pod) *v1alpha1.ElasticJob {
	ownerRef := pod.OwnerReferences[0]
	if ownerRef.Kind != "ElasticJob" {
		return nil
	}
	jobName := ownerRef.Name
	if _, exist := r.CachedJobs[jobName]; !exist {
		namespacedName := types.NamespacedName{
			Name:      jobName,
			Namespace: pod.Namespace,
		}
		job, err := r.fetchElasticJob(namespacedName)
		if job == nil {
			r.Log.Error(err, fmt.Sprintf("Job %s does not exist.", jobName))
			return nil
		}
		r.CachedJobs[jobName] = job
	}
	return r.CachedJobs[jobName]
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

package controllers

import (
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/util/workqueue"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/event"
	"sigs.k8s.io/controller-runtime/pkg/reconcile"
)

var enqueueLog = ctrl.Log.WithName("eventhandler").WithName("EnqueueRequestForPod")

// EnqueueRequestForObject enqueues a Request containing the Name and Namespace of the object that is the source of the Event.
// (e.g. the created / deleted / updated objects Name and Namespace).  handler.EnqueueRequestForObject is used by almost all
// Controllers that have associated Resources (e.g. CRDs) to reconcile the associated Resource.
type EnqueueRequestForPod struct {
	Reconciler *ElasticJobReconciler
}

// Create implements EventHandler.
func (e *EnqueueRequestForPod) Create(evt event.CreateEvent, q workqueue.RateLimitingInterface) {
	if evt.Object == nil {
		enqueueLog.Error(nil, "CreateEvent received with no metadata", "event", evt)
		return
	}
	e.Reconciler.ProcessPodCreateEvent(&evt)
	jobName := getElasticJobNameFromObject(evt.Object)
	q.Add(reconcile.Request{NamespacedName: types.NamespacedName{
		Name:      jobName,
		Namespace: evt.Object.GetNamespace(),
	}})
}

// Update implements EventHandler.
func (e *EnqueueRequestForPod) Update(evt event.UpdateEvent, q workqueue.RateLimitingInterface) {
	e.Reconciler.ProcessPodUpdateEvent(&evt)
	switch {
	case evt.ObjectNew != nil:
		jobName := getElasticJobNameFromObject(evt.ObjectNew)
		q.Add(reconcile.Request{NamespacedName: types.NamespacedName{
			Name:      jobName,
			Namespace: evt.ObjectNew.GetNamespace(),
		}})
	case evt.ObjectOld != nil:
		jobName := getElasticJobNameFromObject(evt.ObjectOld)
		q.Add(reconcile.Request{NamespacedName: types.NamespacedName{
			Name:      jobName,
			Namespace: evt.ObjectOld.GetNamespace(),
		}})
	default:
		enqueueLog.Error(nil, "UpdateEvent received with no metadata", "event", evt)
	}
}

// Delete implements EventHandler.
func (e *EnqueueRequestForPod) Delete(evt event.DeleteEvent, q workqueue.RateLimitingInterface) {
	if evt.Object == nil {
		enqueueLog.Error(nil, "DeleteEvent received with no metadata", "event", evt)
		return
	}
	e.Reconciler.ProcessPodDeleteEvent(&evt)
	jobName := getElasticJobNameFromObject(evt.Object)
	q.Add(reconcile.Request{NamespacedName: types.NamespacedName{
		Name:      jobName,
		Namespace: evt.Object.GetNamespace(),
	}})
}

// Generic implements EventHandler.
func (e *EnqueueRequestForPod) Generic(evt event.GenericEvent, q workqueue.RateLimitingInterface) {
	if evt.Object == nil {
		enqueueLog.Error(nil, "GenericEvent received with no metadata", "event", evt)
		return
	}
	jobName := getElasticJobNameFromObject(evt.Object)
	q.Add(reconcile.Request{NamespacedName: types.NamespacedName{
		Name:      jobName,
		Namespace: evt.Object.GetNamespace(),
	}})
}

func getElasticJobNameFromObject(obj client.Object) string {
	pod := obj.(*corev1.Pod)
	ownerRef := pod.OwnerReferences[0]
	return ownerRef.Name

}

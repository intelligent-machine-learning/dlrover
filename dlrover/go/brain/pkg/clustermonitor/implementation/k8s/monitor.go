package k8s

import (
	"fmt"
	"google.golang.org/grpc"
	"k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/tools/cache"
	"time"
)

const (
	Pod      = "Pod"
	Training = "Training"
)

// Monitor is an agent to persist resources to a database.
type Monitor struct {
	informerSyncers []cache.InformerSynced
	monitorWorkers  []worker.MonitorWorkerInterface
}

// NewMonitorAgent returns a new monitor agent.
func NewMonitorAgent(podInformerFactory kubeinformers.SharedInformerFactory, workflowInformerFactory argoinformers.SharedInformerFactory, trainingInformerFactory externalversions.SharedInformerFactory, notebookInformerFactory externalversions.SharedInformerFactory, grpcConn *grpc.ClientConn, config *config.ConfigSpec, time utils.TimeInterface) *MonitorAgent {
	var informerSyncers []cache.InformerSynced
	var monitorWorkers []worker.MonitorWorkerInterface

	informerSyncers, monitorWorkers = appendWorkflowSyncedAndWorker(workflowInformerFactory, time, grpcConn, config.Cluster, informerSyncers, monitorWorkers)
	informerSyncers, monitorWorkers = appendPodSyncedAndWorker(podInformerFactory, trainingInformerFactory, notebookInformerFactory, time, grpcConn, config.Cluster, informerSyncers, monitorWorkers)
	informerSyncers, monitorWorkers = appendDryrunPodSyncedAndWorker(podInformerFactory, trainingInformerFactory, notebookInformerFactory, time, grpcConn, config.Cluster, informerSyncers, monitorWorkers)
	informerSyncers, monitorWorkers = appendTrainingSyncedAndWorker(trainingInformerFactory, time, grpcConn, config.Cluster, informerSyncers, monitorWorkers)
	informerSyncers, monitorWorkers = appendNotebookSyncedAndWorker(notebookInformerFactory, time, grpcConn, config.Cluster, informerSyncers, monitorWorkers)

	agent := &MonitorAgent{
		informerSyncers: informerSyncers,
		monitorWorkers:  monitorWorkers,
	}

	log.Info("Setting up event handlers")
	return agent
}

func appendTrainingSyncedAndWorker(
	trainingInformerFactory externalversions.SharedInformerFactory,
	time utils.TimeInterface,
	grpcConn *grpc.ClientConn,
	cluster string,
	informerSyncers []cache.InformerSynced,
	monitorWorkers []worker.MonitorWorkerInterface) ([]cache.InformerSynced, []worker.MonitorWorkerInterface) {
	if trainingInformerFactory == nil {
		return informerSyncers, monitorWorkers
	}
	trainingInformer := trainingInformerFactory.Jobs().V1beta1().Trainings()
	trainingClient := client.NewTrainingClient(trainingInformer)
	grpcJobClient := v1.NewJobServiceClient(grpcConn)
	grpcTaskGroupClient := v1.NewTaskGroupServiceClient(grpcConn)
	trainingSaver := worker.NewTrainingSaver(trainingClient, cluster, grpcJobClient, grpcTaskGroupClient)
	trainingWorker := worker.NewGeneralMonitorWorker(time, Training,
		trainingInformer.Informer(), true, trainingSaver)
	informerSyncers = append(informerSyncers, trainingClient.HasSynced())
	monitorWorkers = append(monitorWorkers, trainingWorker)
	log.Info("add trainingWorker")
	return informerSyncers, monitorWorkers
}

func appendNotebookSyncedAndWorker(
	notebookInformerFactory externalversions.SharedInformerFactory,
	time utils.TimeInterface,
	grpcConn *grpc.ClientConn,
	cluster string,
	informerSyncers []cache.InformerSynced,
	monitorWorkers []worker.MonitorWorkerInterface) ([]cache.InformerSynced, []worker.MonitorWorkerInterface) {
	if notebookInformerFactory == nil {
		return informerSyncers, monitorWorkers
	}
	notebookInformer := notebookInformerFactory.Jobs().V1beta1().Notebooks()
	notebookClient := client.NewNotebookClient(notebookInformer)
	grpcJobClient := v1.NewJobServiceClient(grpcConn)
	grpcTaskGroupClient := v1.NewTaskGroupServiceClient(grpcConn)
	notebookSaver := worker.NewNotebookSaver(notebookClient, cluster, grpcJobClient, grpcTaskGroupClient)
	notebookWorker := worker.NewGeneralMonitorWorker(time, Notebook,
		notebookInformer.Informer(), true, notebookSaver)
	informerSyncers = append(informerSyncers, notebookClient.HasSynced())
	monitorWorkers = append(monitorWorkers, notebookWorker)
	log.Info("add notebookWorker")
	return informerSyncers, monitorWorkers

}

func appendPodSyncedAndWorker(podInformerFactory kubeinformers.SharedInformerFactory, trainingInformerFactory externalversions.SharedInformerFactory, notebookInformerFactory externalversions.SharedInformerFactory, time utils.TimeInterface, grpcConn *grpc.ClientConn, cluster string, informerSyncers []cache.InformerSynced, monitorWorkers []worker.MonitorWorkerInterface) ([]cache.InformerSynced, []worker.MonitorWorkerInterface) {
	if podInformerFactory == nil {
		return informerSyncers, monitorWorkers
	}
	podInformer := podInformerFactory.Core().V1().Pods()
	podClient := client.NewPodClient(podInformer)
	if trainingInformerFactory == nil {
		return informerSyncers, monitorWorkers
	}
	trainingInformer := trainingInformerFactory.Jobs().V1beta1().Trainings()
	trainingClient := client.NewTrainingClient(trainingInformer)
	notebookInformer := notebookInformerFactory.Jobs().V1beta1().Notebooks()
	notebookClient := client.NewNotebookClient(notebookInformer)
	grpcClient := v1.NewTaskServiceClient(grpcConn)
	metricsClient := v1.NewGeneralMetricsServiceClient(grpcConn)
	taskSaver := worker.NewTaskSaver(podClient, trainingClient, notebookClient, cluster, grpcClient, metricsClient)
	podWorker := worker.NewGeneralMonitorWorker(time, Pod,
		podInformer.Informer(), true, taskSaver)
	informerSyncers = append(informerSyncers, podClient.HasSynced())
	monitorWorkers = append(monitorWorkers, podWorker)
	log.Info("add podWorker")
	return informerSyncers, monitorWorkers
}

func appendDryrunPodSyncedAndWorker(podInformerFactory kubeinformers.SharedInformerFactory,
	trainingInformerFactory externalversions.SharedInformerFactory,
	notebookInformerFactory externalversions.SharedInformerFactory,
	time utils.TimeInterface, grpcConn *grpc.ClientConn,
	cluster string, informerSyncers []cache.InformerSynced, monitorWorkers []worker.MonitorWorkerInterface) ([]cache.InformerSynced, []worker.MonitorWorkerInterface) {
	if podInformerFactory == nil {
		return informerSyncers, monitorWorkers
	}
	podInformer := podInformerFactory.Core().V1().Pods()
	podClient := client.NewPodClient(podInformer)
	if trainingInformerFactory == nil {
		return informerSyncers, monitorWorkers
	}
	trainingInformer := trainingInformerFactory.Jobs().V1beta1().Trainings()
	trainingClient := client.NewTrainingClient(trainingInformer)
	notebookInformer := notebookInformerFactory.Jobs().V1beta1().Notebooks()
	notebookClient := client.NewNotebookClient(notebookInformer)
	grpcDryrunClient := v1.NewDryrunTaskServiceClient(grpcConn)
	dryrunTaskSaver := worker.NewDryrunTaskSaver(podClient, trainingClient, notebookClient, cluster, grpcDryrunClient)
	dryrunPodWorker := worker.NewGeneralMonitorWorker(time, "DryrunPod",
		podInformer.Informer(), true, dryrunTaskSaver)
	informerSyncers = append(informerSyncers, podClient.HasSynced())
	monitorWorkers = append(monitorWorkers, dryrunPodWorker)
	log.Info("add dryrunPodWorker")
	return informerSyncers, monitorWorkers
}

func appendWorkflowSyncedAndWorker(
	workflowInformerFactory argoinformers.SharedInformerFactory,
	time utils.TimeInterface,
	grpcConn *grpc.ClientConn,
	cluster string,
	informerSyncers []cache.InformerSynced,
	monitorWorkers []worker.MonitorWorkerInterface) ([]cache.InformerSynced, []worker.MonitorWorkerInterface) {
	if workflowInformerFactory == nil {
		return informerSyncers, monitorWorkers
	}
	workflowInformer := workflowInformerFactory.Argoproj().V1alpha1().Workflows()
	workflowClient := client.NewWorkflowClient(workflowInformer)
	grpcClient := v1.NewWorkflowServiceClient(grpcConn)
	workflowSaver := worker.NewWorkflowSaver(workflowClient, grpcClient, cluster)
	workflowWorker := worker.NewGeneralMonitorWorker(time, Workflow,
		workflowInformer.Informer(), true, workflowSaver)
	informerSyncers = append(informerSyncers, workflowClient.HasSynced())
	monitorWorkers = append(monitorWorkers, workflowWorker)
	log.Info("add workflowWorker")
	return informerSyncers, monitorWorkers
}

// Run will set up the event handlers for types we are interested in, as well
// as syncing informer caches and starting workers. It will block until stopCh
// is closed, at which point it will shutdown the workqueue and wait for
// workers to finish processing their current work items.
func (p *MonitorAgent) Run(threadiness int, stopCh <-chan struct{}) error {
	defer runtime.HandleCrash()
	for _, monitorWorker := range p.monitorWorkers {
		defer monitorWorker.Shutdown()
	}

	// Start the informer factories to begin populating the informer caches
	log.Info("Starting the monitor agent")

	// Wait for the caches to be synced before starting workers
	log.Info("Waiting for informer caches to sync, number is ", len(p.informerSyncers))

	if ok := cache.WaitForCacheSync(stopCh, p.informerSyncers...); !ok {
		log.Error("Failed to wait for caches to sync")
		return fmt.Errorf("failed to wait for caches to sync")
	}

	// Launch multiple workers to process
	log.Info("Starting workers")
	for i := 0; i < threadiness; i++ {
		for _, monitorWorker := range p.monitorWorkers {
			go wait.Until(monitorWorker.RunWorker, time.Second, stopCh)
		}
	}

	log.Info("Started workers")
	log.Info("Wait for shut down")
	<-stopCh
	log.Info("Shutting down workers")
	return nil
}

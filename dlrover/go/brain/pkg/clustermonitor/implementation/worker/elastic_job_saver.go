package worker

import (
	"fmt"
	"github.com/intelligent-machine-learning/easydl/brain/pkg/clustermonitor/implementation/client"
	elasticv1alpha1 "github.com/intelligent-machine-learning/easydl/operator/api/v1alpha1"
	"strconv"
	"time"
)

const (
	Master      = "master"
	PS          = "ps"
	Worker      = "worker"
	Evaluator   = "evaluator"
	trainingGVK = "jobs.kubemaker.alipay.net/v1beta1, Kind=Training"
)

var trainingStateMap = map[jobsv1beta1.TrainingState]common.RunningStatus{
	jobsv1beta1.TrainingStatePending:  common.RunningStatus_Pending,
	jobsv1beta1.TrainingStateRunning:  common.RunningStatus_Running,
	jobsv1beta1.TrainingStateComplete: common.RunningStatus_Succeeded,
	jobsv1beta1.TrainingStateError:    common.RunningStatus_Failed,
	jobsv1beta1.TrainingStateUnknown:  common.RunningStatus_Unknown,
}

type TrainingSaver struct {
	job                    *elasticv1alpha1.ElasticJob
	client                 client.TrainingClientInterface
	cluster                string
	jobServiceClient       v1.JobServiceClient
	taskGroupServiceClient v1.TaskGroupServiceClient
}

func NewTrainingSaver(client client.TrainingClientInterface, cluster string, jobServiceClient v1.JobServiceClient, taskGroupServiceClient v1.TaskGroupServiceClient) *TrainingSaver {
	return &TrainingSaver{
		client:                 client,
		cluster:                cluster,
		jobServiceClient:       jobServiceClient,
		taskGroupServiceClient: taskGroupServiceClient,
	}
}

// Process the training job
func (s *TrainingSaver) Save(key string, namespace string, name string, nowEpoch int64) error {
	// Get the Training with this namespace/name
	training, err := s.client.Get(namespace, name)
	if utils.HasCustomCode(err, utils.CUSTOM_CODE_NOT_FOUND) {
		// Permanent failure. Mark the deletion time.
		response, err2 := s.jobServiceClient.List(context.Background(), &v1.ListJobRequest{Namespace: namespace, Name: name})
		if err2 != nil {
			log.Errorf("Failed to list job %s/%s: %v", namespace, name, err2)
		} else if response.Status != common.ResponseStatus_SUCCESS {
			log.Errorf("Failed to list job %s/%s: %v", namespace, name, response.Error)
		} else if len(response.Jobs) == 0 {
			log.Errorf("Unable to find any job matching namespace=%s, name=%s, so we skip updating RemovedAt field", namespace, name)
		} else {
			savedJob := response.Jobs[len(response.Jobs)-1]
			savedJob.RemovedAt = nowEpoch
			savedJob.Status = common.RunningStatus_Removed
			if savedJob.FinishedAt == 0 {
				savedJob.FinishedAt = nowEpoch
			}
			response, err3 := s.jobServiceClient.Update(context.Background(), savedJob)
			if err3 != nil {
				log.Errorf("Failed to update job %s/%s RemovedAt field: %v", namespace, name, err3)
			} else if response.Status != common.ResponseStatus_SUCCESS {
				log.Errorf("Failed to update job %s/%s RemovedAt field: %s", namespace, name, response.Error)
			}
			// TODO(fangliu): update task groups
		}
		return utils.NewCustomError(err, utils.CUSTOM_CODE_PERMANENT,
			"Training (%s) in work queue no longer exists: %v", key, err)
	}
	if err != nil {
		// Transient failure, we will retry
		return utils.NewCustomError(err, utils.CUSTOM_CODE_TRANSIENT,
			"Training (%s): transient failure: %v", key, err)
	}

	log.Infof("Updating training & task groups %s/%s to the metadata service", namespace, name)
	ctx := context.Background()
	job := convertTrainingToRPCJob(training, s.cluster)
	jobResp, err := s.jobServiceClient.Update(ctx, job)
	var errMsg string
	if err != nil {
		errMsg = fmt.Sprintf("Failed to update job %s/%s: %v", namespace, name, err)
	} else if jobResp.Status != common.ResponseStatus_SUCCESS {
		errMsg = fmt.Sprintf("Failed to update job %s/%s, and the metadata service returns error with message: %s",
			namespace, name, jobResp.Error)
	}
	if len(errMsg) > 0 {
		log.Error(errMsg)
		return utils.NewCustomError(err, utils.CUSTOM_CODE_METADATA_SERVICE, errMsg)
	}

	taskGroups := generateRPCTaskGroups(training, job, s.cluster)
	// TODO(fangliu): Batch upsert
	for _, taskGroup := range taskGroups {
		taskGroupResp, err := s.taskGroupServiceClient.Upsert(ctx, taskGroup)
		var errMsg string
		if err != nil {
			errMsg = fmt.Sprintf("Failed to update task group %s/%s.%s: %v", namespace, name, taskGroup.Name, err)
		} else if taskGroupResp.Status != common.ResponseStatus_SUCCESS {
			errMsg = fmt.Sprintf("Failed to update task group %s/%s.%s, and the metadata service returns error with message: %v",
				namespace, name, taskGroup.Name, taskGroupResp.Error)
		}
		if len(errMsg) > 0 {
			log.Error(errMsg)
			return utils.NewCustomError(err, utils.CUSTOM_CODE_METADATA_SERVICE, errMsg)
		}
	}

	return nil
}

func convertTrainingToRPCJob(training *jobsv1beta1.Training, cluster string) *v1.Job {
	content, err := json.Marshal(training)
	if err != nil {
		log.Errorf("Failed to marshal training %s: %v", training.GetName(), err.Error())
	}

	now := time.Now().Unix()
	user := utils.GetUser(training.Labels)
	createdAt := training.ObjectMeta.CreationTimestamp.Unix()
	app := utils.GetAppName(training.Labels)
	if app == "" {
		app = "kmaker"
	}
	job := &v1.Job{
		Name:             training.Name,
		Namespace:        training.Namespace,
		Cluster:          cluster,
		Type:             common.JobType_Training,
		Status:           trainingStateToRunningStatus(training.Status.State),
		OptConfig:        utils.GetOptConfig(training.Annotations),
		GroupVersionKind: trainingGVK, // FIXME(fangliu): Some submitted trainings doesn't have GVK
		User:             user,
		ResourcePool:     utils.GetPool(training.Labels),
		App:              app,
		Content:          string(content),
		CreatedAt:        createdAt,
		OptMode:          utils.GetOptMode(training.Labels),
		Uuid:             GenerateTrainingJobID(training),
		ProfileId:        user,                       // Training treats user as the profile ID
		Version:          strconv.FormatInt(now, 10), // For now training doesn't support HPA, set it to current timestamp is enough
	}

	if training.Status.StartRunTime != nil {
		job.StartedAt = training.Status.StartRunTime.Unix()
	} else if training.Status.State == jobsv1beta1.TrainingStateRunning {
		job.StartedAt = now
	}
	if training.Status.CompletionTime != nil {
		job.FinishedAt = training.Status.CompletionTime.Unix()
	} else if training.Status.State == jobsv1beta1.TrainingStateComplete {
		job.FinishedAt = now
	}
	return job
}

func trainingStateToRunningStatus(state jobsv1beta1.TrainingState) common.RunningStatus {
	if status, ok := trainingStateMap[state]; ok {
		return status
	}
	return common.RunningStatus_UnspecifiedStatus
}

// generateRPCTaskGroups generates task groups from training.
// For now they will not change after created, so generate once for all.
func generateRPCTaskGroups(training *jobsv1beta1.Training, job *v1.Job, cluster string) []*v1.TaskGroup {
	taskGroups := make([]*v1.TaskGroup, 0)
	hboEnabled := utils.IsHBOEnabled(training.Labels)
	currentResourcesMap, err := utils.GetVPATaskResourcesMap(training.Annotations)
	if err != nil {
		log.Errorf("Error read current resources from training: %v", err)
	}
	// FIXME(fangliu): ElasticDL eliminates default master spec in crd, refill it here. Should move this to a webhook in the future.
	if training.Spec.Runtime == jobsv1beta1.JobTypeElasticDL && training.Spec.Master == nil {
		one := int32(1)
		training.Spec.Master = &jobsv1beta1.TaskSpec{
			Count: &one,
			Type:  jobsv1beta1.TaskTypeCustom,
			CustomType: &jobsv1beta1.ContainerTypeSpec{
				CPU:    4.0,
				Memory: 2048,
			},
		}
	}
	taskSpecsMap := map[string]*jobsv1beta1.TaskSpec{
		Master:    training.Spec.Master,
		PS:        training.Spec.ParameterServer,
		Worker:    training.Spec.Worker,
		Evaluator: training.Spec.Evaluator,
	}
	for role, taskSpec := range taskSpecsMap {
		if taskSpec == nil || taskSpec.Count == nil || *taskSpec.Count == 0 {
			continue
		}
		taskSpecCount, taskSpecResources := *taskSpec.Count, utils.GenerateResourcesFromTaskSpec(taskSpec)

		// By default get user requested resources from spec
		taskNumRequested, resourcesRequested := taskSpecCount, taskSpecResources
		// And set HBO resources to zero-values
		hboTaskNum, hboResources := int32(0), &common.Resources{}
		// Unless HBO is enabled and successfully applied
		if hboEnabled {
			if taskNumRequestedBeforeHBO, resourcesRequestedBeforeHBO, err := utils.GetUserRequestedResourcesBeforeHBO(training.Annotations, role); err == nil {
				taskNumRequested, resourcesRequested = taskNumRequestedBeforeHBO, resourcesRequestedBeforeHBO
				hboTaskNum, hboResources = taskSpecCount, taskSpecResources
			}
		}

		// By default get current resources from spec
		currentTaskNum, currentResources := taskSpecCount, taskSpecResources
		if optPlan, ok := currentResourcesMap[role]; ok && optPlan != nil {
			// Unless VPA has been successfully applied
			// Desired currentTaskNum should be what VPA wanted
			currentTaskNum, currentResources = optPlan.Replicas, optPlan.Resources
		}

		// Optimizer framework relies on started_at & finished_at to judge if task group is running,
		// then checks whether CurrentTaskNum == RunningTaskNum. It'll begin to optimize only when all matched.
		// Here we set times to job times, and CurrentTaskNum to requested task num for simplicity.
		// As a result, optimizer framework will do optimization only when the job is running and all tasks in this group is running.
		taskGroup := &v1.TaskGroup{
			JobUuid:           job.Uuid,
			JobName:           job.Name,
			JobType:           job.Type,
			Namespace:         job.Namespace,
			Cluster:           cluster,
			Name:              role,
			ProfileId:         job.ProfileId,
			User:              job.User,
			TaskNumRequested:  taskNumRequested,
			ResourceRequested: resourcesRequested,
			HboTaskNum:        hboTaskNum,
			HboResources:      hboResources,
			CurrentTaskNum:    currentTaskNum,
			CurrentResources:  currentResources,
			CreatedAt:         job.CreatedAt,
			StartedAt:         job.StartedAt,
			FinishedAt:        job.FinishedAt,
			RemovedAt:         job.RemovedAt,
		}
		taskGroups = append(taskGroups, taskGroup)
	}
	return taskGroups
}

func GenerateTrainingJobID(training *jobsv1beta1.Training) string {
	return trainingJobIDPrefix + string(training.GetUID())
}

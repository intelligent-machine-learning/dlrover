package client

import (
	elasticv1alpha1 "github.com/intelligent-machine-learning/easydl/operator/api/v1alpha1"
	"k8s.io/client-go/tools/cache"
)

type ElasticJobClientInterface interface {
	Get(namespace string, name string) (*elasticv1alpha1.ElasticJob, error)
}

// TrainingClient is a client to call the Training API.
type ElasticJobClient struct {
	informer jobsv1beta1informers.TrainingInformer
}

// NewTrainingClient creates an instance of the TrainingClient.
func NewTrainingClient(informer jobsv1beta1informers.TrainingInformer) *TrainingClient {
	return &TrainingClient{informer: informer}
}

// AddEventHandler adds an event handler.
func (c *TrainingClient) AddEventHandler(funcs *cache.ResourceEventHandlerFuncs) {
	c.informer.Informer().AddEventHandler(funcs)
}

// HasSynced returns true if the shared informer's store has synced.
func (c *TrainingClient) HasSynced() func() bool {
	return c.informer.Informer().HasSynced
}

// Get returns a Training, given a namespace and name.
func (c *TrainingClient) Get(namespace string, name string) (
	t *jobsv1beta1.Training, err error) {
	training, err := c.informer.Lister().Trainings(namespace).Get(name)
	if err != nil {
		var code utils.CustomCode
		if utils.IsNotFound(err) {
			code = utils.CUSTOM_CODE_NOT_FOUND
		} else {
			code = utils.CUSTOM_CODE_GENERIC
		}
		return nil, utils.NewCustomError(err, code,
			"Error retrieving training (%s) in namespace (%s): %v", name, namespace, err)
	}
	return training, nil
}

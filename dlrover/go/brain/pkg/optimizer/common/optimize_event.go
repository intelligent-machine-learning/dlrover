package common

import (
	"github.com/intelligent-machine-learning/easydl/brain/pkg/common"
	"github.com/intelligent-machine-learning/easydl/brain/pkg/optimizer/config"
)

// OptimizeEvent is the struct of optimize event
type OptimizeEvent struct {
	Type           string
	ProcessorName  string
	DataStoreName  string
	Jobs           []*common.JobMeta
	Conf           *config.OptimizerConfig
	CustomizedData map[string]string
}

package api

import (
	"context"
	"github.com/intelligent-machine-learning/easydl/brain/pkg/common"
)

type ClusterMonitor interface {
	Start(ctx context.Context, errReporter common.ErrorReporter) error
	Stop() error
}

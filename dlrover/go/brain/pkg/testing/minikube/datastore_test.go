package minikube

import (
	"github.com/intelligent-machine-learning/easydl/brain/pkg/config"
	dsimpl "github.com/intelligent-machine-learning/easydl/brain/pkg/datastore/implementation"
	"github.com/intelligent-machine-learning/easydl/brain/pkg/datastore/recorder/mysql"
	pb "github.com/intelligent-machine-learning/easydl/brain/pkg/proto"
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestDatastore(t *testing.T) {
	err := testMysqlConnection()
	assert.NoError(t, err)
}

func testMysqlConnection() error {
	conf := config.NewEmptyConfig()
	conf.Set(config.DBUser, "root")
	conf.Set(config.DBPassword, "root")
	conf.Set(config.DBEngineType, "mysql")
	//conf.Set(config.DBURL, "mysql.dlrover.svc.cluster.local:3306")
	conf.Set(config.DBURL, "tcp(127.0.0.1:53254)/dlrover?parseTime=true&interpolateParams=true&loc=Local")
	dbClient := mysql.NewClient(conf)

	baseDatastore := &dsimpl.BaseDataStore{
		Client: dbClient,
	}

	jobMetrics := &pb.JobMetrics{
		JobMeta: &pb.JobMeta{
			Uuid: "testing-job-uuid",
			Name: "testing-job",
		},
		MetricsType: pb.MetricsType_Job_Exit_Reason,
		Metrics: &pb.JobMetrics_JobExitReason{
			JobExitReason: "oom",
		},
	}

	return baseDatastore.PersistData(nil, jobMetrics, nil)
}

package server

import (
	"context"
	"fmt"
	"github.com/golang/protobuf/ptypes/empty"
	"github.com/intelligent-machine-learning/easydl/brain/pkg/common"
	"github.com/intelligent-machine-learning/easydl/brain/pkg/config"
	datastoreapi "github.com/intelligent-machine-learning/easydl/brain/pkg/datastore/api"
	pb "github.com/intelligent-machine-learning/easydl/brain/pkg/proto"
)

// BrainServer is the interface of DLRover Brain
type BrainServer struct {
	pb.UnimplementedBrainServer
}

// NewBrainServer creates an EasyDLServer instance
func NewBrainServer(conf *config.Config) (*BrainServer, error) {
	return &BrainServer{}, nil
}

// Run starts the server
func (s *BrainServer) Run(ctx context.Context, errReporter common.ErrorReporter) error {
	return nil
}

// ELJobConfigUpdateNotify updates the EDL job config when to observe an update
func (s *EasyDLServer) ELJobConfigUpdateNotify(newConf *config.Config) error {
	log.Infof("EasyDL job new conf: %v", newConf)
	s.elasticDLJobConf = newConf
	return nil
}

// PersistMetrics persists job metrics to data store
func (s *EasyDLServer) PersistMetrics(ctx context.Context, in *pb.JobMetrics) (*empty.Empty, error) {
	dataStore, err := s.getDataStore(in.DataStore)
	if err != nil {
		return nil, err
	}
	err = dataStore.PersistMetrics(nil, in, nil)
	if err != nil {
		return nil, err
	}
	return &empty.Empty{}, nil
}

func (s *EasyDLServer) getDataStore(name string) (datastoreapi.DataStore, error) {
	_, found := s.dataStores[name]
	if !found {
		dataStore, err := s.dataStoreManager.CreateDataStore(name)
		if err != nil {
			return nil, err
		}
		s.dataStores[name] = dataStore
	}
	return s.dataStores[name], nil
}

// Optimize returns the initial resource of a job.
func (s *EasyDLServer) Optimize(ctx context.Context, in *pb.OptimizeRequest) (*pb.OptimizeResponse, error) {
	plans, err := s.optimizerManager.Optimize(in)
	if err != nil {
		errReason := fmt.Sprintf("Fail to optimize request %v: %v", in, err)
		log.Errorf(errReason)
		return &pb.OptimizeResponse{
			Response: &pb.Response{Success: false, Reason: errReason},
		}, err
	}
	if plans != nil {
		jobs := in.GetJobs()
		for i, plan := range plans {
			jobOptimization := &pb.JobOptimization{
				Config: in.GetConfig(),
				Plan:   plan,
			}
			jobMetrics := &pb.JobMetrics{
				JobMeta: &pb.JobMeta{
					Uuid: jobs[i].Uid,
				},
			}
			dataStore, err := s.getDataStore(in.Config.DataStore)
			if err != nil {
				log.Errorf("fail to get data store %s: %v", in.Config.DataStore, err)
				continue
			}

			err = dataStore.PersistMetrics(nil, jobMetrics, jobOptimization)
			if err != nil {
				log.Errorf("fail to persistence job %s optimization: %v", jobs[i].Uid, err)
			}
		}

	}

	return &pb.OptimizeResponse{
		Response: &pb.Response{
			Success: true,
		},
		JobOptimizePlans: plans,
	}, nil
}

// GetConfig gets the config for ElasticDL job
func (s *EasyDLServer) GetConfig(ctx context.Context, in *pb.ConfigRequest) (*pb.ConfigResponse, error) {
	configVal := s.elasticDLJobConf.GetString(in.ConfigKey)
	if len(configVal) == 0 {
		return &pb.ConfigResponse{
			Response: &pb.Response{
				Success: false,
			},
		}, nil
	}

	return &pb.ConfigResponse{
		Response: &pb.Response{
			Success: true,
		},
		ConfigValue: configVal,
	}, nil
}

// GetJobMetrics returns a job metrics
func (s *EasyDLServer) GetJobMetrics(ctx context.Context, in *pb.JobMetricsRequest) (*pb.JobMetricsResponse, error) {
	dataStore, err := s.getDataStore(datastoreimpl.DataStoreElasticDL)
	if err != nil {
		log.Errorf("fail to get data store %s", datastoreimpl.DataStoreElasticDL)
		return &pb.JobMetricsResponse{
			Response: &pb.Response{
				Success: false,
				Reason:  "fail to get data store",
			},
		}, nil
	}

	cond := &datastoreapi.Condition{
		Type: common.TypeGetDataGetJobMetrics,
		Extra: &cougardb.JobMetricsCondition{
			JobUUID: in.JobUuid,
		},
	}
	jobMetrics := &cougardb.JobMetrics{}
	err = dataStore.GetData(cond, jobMetrics)
	if err != nil {
		log.Errorf("Fail to get job_metrics for %s: %v", in.JobUuid, err)
		return &pb.JobMetricsResponse{
			Response: &pb.Response{
				Success: false,
				Reason:  "fail to get job metrics from db",
			},
		}, err
	}

	str, err := json.Marshal(jobMetrics)
	if err != nil {
		log.Errorf("fail to marshal job metrics of %s: %v", in.JobUuid, err)
		return &pb.JobMetricsResponse{
			Response: &pb.Response{
				Success: false,
				Reason:  "fail to marshal job metrics",
			},
		}, err
	}

	return &pb.JobMetricsResponse{
		Response: &pb.Response{
			Success: true,
		},
		JobMetrics: string(str),
	}, nil
}

// ProcessKubeWatchEvent assigns KubeWatchEvent to registered administrators for processing
func (s *EasyDLServer) ProcessKubeWatchEvent(ctx context.Context, event *pb.KubeWatchEvent) (*pb.Response, error) {
	for _, adminName := range event.Admins {
		customizedData := make(map[string]string)
		if event.CustomizedData != nil {
			for k, v := range event.CustomizedData {
				customizedData[k] = v
			}
		}

		adminEvent := &braincommon.AdministratorEvent{
			EventType:      event.Type,
			Jobs:           utils.ConvertPBJobMetaArrayToJobMetaArray(event.Jobs),
			CustomizedData: customizedData,
		}

		err := s.adminManager.ProcessEvent(ctx, adminName, adminEvent)
		if err != nil {
			log.Errorf("administrator %s fail to process event %v: %v", adminName, adminEvent, err)
		}
	}
	return &pb.Response{
		Success: true,
	}, nil
}

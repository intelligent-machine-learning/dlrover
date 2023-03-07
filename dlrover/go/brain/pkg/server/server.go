// Copyright 2022 The DLRover Authors. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package server

import (
	"context"
	"encoding/json"
	"fmt"
	log "github.com/golang/glog"
	"github.com/golang/protobuf/ptypes/empty"
	"github.com/intelligent-machine-learning/easydl/brain/pkg/common"
	"github.com/intelligent-machine-learning/easydl/brain/pkg/config"
	"github.com/intelligent-machine-learning/easydl/brain/pkg/datastore"
	datastoreapi "github.com/intelligent-machine-learning/easydl/brain/pkg/datastore/api"
	dsimpl "github.com/intelligent-machine-learning/easydl/brain/pkg/datastore/implementation"
	"github.com/intelligent-machine-learning/easydl/brain/pkg/datastore/recorder/mysql"
	"github.com/intelligent-machine-learning/easydl/brain/pkg/optimizer"
	pb "github.com/intelligent-machine-learning/easydl/brain/pkg/proto"
	"k8s.io/client-go/kubernetes"
)

const (
	defaultDataStoreName = dsimpl.BaseDataStoreName
	logName              = "brain-server"
)

// BrainServer is the interface of DLRover Brain
type BrainServer struct {
	pb.UnimplementedBrainServer

	kubeClientSet kubernetes.Interface

	conf          *config.Config
	configManager *config.Manager

	dsManager *datastore.Manager
	manager   *optimizer.Manager
}

// NewBrainServer creates an EasyDLServer instance
func NewBrainServer(conf *config.Config) (*BrainServer, error) {
	namespace := conf.GetString(config.Namespace)
	configMapName := conf.GetString(config.BrainServerConfigMapName)
	configMapKey := conf.GetString(config.BrainServerConfigMapKey)
	kubeClientSet := conf.GetKubeClientInterface()

	return &BrainServer{
		configManager: config.NewManager(namespace, configMapName, configMapKey, kubeClientSet),
		kubeClientSet: kubeClientSet,
	}, nil
}

// Run starts the server
func (s *BrainServer) Run(ctx context.Context, errReporter common.ErrorReporter) error {
	log.Infof("start to run brain server")
	err := s.configManager.Run(ctx, errReporter)
	if err != nil {
		err = fmt.Errorf("[%s] failed to initialize config manager: %v", logName, err)
		log.Error(err)
		return err
	}
	s.conf, err = s.configManager.GetConfig()
	if err != nil {
		log.Errorf("[%s] fail to get brain server config: %v", logName, err)
		return err
	}
	s.conf.Set(config.KubeClientInterface, s.kubeClientSet)

	s.dsManager = datastore.NewManager(s.conf)
	if err = s.dsManager.Run(ctx, errReporter); err != nil {
		log.Errorf("[%s] fail to run the data store manager: %v", logName, err)
		return err
	}

	s.manager = optimizer.NewManager(s.conf)
	if err = s.manager.Run(ctx, errReporter); err != nil {
		log.Errorf("[%s] fail to run the manager: %v", logName, err)
		return err
	}
	return nil
}

// PersistMetrics persists job metrics to data store
func (s *BrainServer) PersistMetrics(ctx context.Context, in *pb.JobMetrics) (*empty.Empty, error) {
	dataStore, err := s.dsManager.CreateDataStore(in.DataStore)
	if err != nil {
		return nil, err
	}
	err = dataStore.PersistData(nil, in, nil)
	if err != nil {
		return nil, err
	}
	return &empty.Empty{}, nil
}

// Optimize returns the initial resource of a job.
func (s *BrainServer) Optimize(ctx context.Context, in *pb.OptimizeRequest) (*pb.OptimizeResponse, error) {
	log.Infof("Receive optimize request: %v", in)
	plans, err := s.manager.ProcessOptimizeRequest(ctx, in)
	if err != nil {
		log.Errorf("[%s] fail to process request %v: %v", logName, in, err)
		return &pb.OptimizeResponse{
			Response: &pb.Response{
				Success: false,
			},
		}, err
	}

	return &pb.OptimizeResponse{
		Response: &pb.Response{
			Success: true,
		},
		JobOptimizePlans: plans,
	}, nil
}

// GetJobMetrics returns a job metrics
func (s *BrainServer) GetJobMetrics(ctx context.Context, in *pb.JobMetricsRequest) (*pb.JobMetricsResponse, error) {
	dataStore, err := s.dsManager.CreateDataStore(defaultDataStoreName)
	if err != nil {
		log.Errorf("fail to get data store %s", defaultDataStoreName)
		return &pb.JobMetricsResponse{
			Response: &pb.Response{
				Success: false,
				Reason:  "fail to get data store",
			},
		}, nil
	}

	cond := &datastoreapi.Condition{
		Type: common.TypeGetDataGetJobMetrics,
		Extra: &mysql.JobMetricsCondition{
			UID: in.JobUuid,
		},
	}
	jobMetrics := &mysql.JobMetrics{}
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

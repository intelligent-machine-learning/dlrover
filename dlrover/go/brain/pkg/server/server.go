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
	log "github.com/golang/glog"
	"github.com/golang/protobuf/ptypes/empty"
	"github.com/intelligent-machine-learning/easydl/brain/pkg/common"
	"github.com/intelligent-machine-learning/easydl/brain/pkg/config"
	"github.com/intelligent-machine-learning/easydl/brain/pkg/datastore"
	datastoreapi "github.com/intelligent-machine-learning/easydl/brain/pkg/datastore/api"
	dsimpl "github.com/intelligent-machine-learning/easydl/brain/pkg/datastore/implementation"
	"github.com/intelligent-machine-learning/easydl/brain/pkg/datastore/recorder/mysql"
	pb "github.com/intelligent-machine-learning/easydl/brain/pkg/proto"
)

const (
	defaultDataStoreName = dsimpl.BaseDataStoreName
)

// BrainServer is the interface of DLRover Brain
type BrainServer struct {
	pb.UnimplementedBrainServer
	dsManager  *datastore.Manager
	dataStores map[string]datastoreapi.DataStore
}

// NewBrainServer creates an EasyDLServer instance
func NewBrainServer(conf *config.Config) (*BrainServer, error) {
	dsManager := datastore.NewManager(conf)
	return &BrainServer{
		dsManager:  dsManager,
		dataStores: make(map[string]datastoreapi.DataStore),
	}, nil
}

// Run starts the server
func (s *BrainServer) Run(ctx context.Context, errReporter common.ErrorReporter) error {
	return s.dsManager.Run(ctx, errReporter)
}

// PersistMetrics persists job metrics to data store
func (s *BrainServer) PersistMetrics(ctx context.Context, in *pb.JobMetrics) (*empty.Empty, error) {
	dataStore, err := s.getDataStore(in.DataStore)
	if err != nil {
		return nil, err
	}
	err = dataStore.PersistData(nil, in, nil)
	if err != nil {
		return nil, err
	}
	return &empty.Empty{}, nil
}

func (s *BrainServer) getDataStore(name string) (datastoreapi.DataStore, error) {
	_, found := s.dataStores[name]
	if !found {
		dataStore, err := s.dsManager.CreateDataStore(name)
		if err != nil {
			return nil, err
		}
		s.dataStores[name] = dataStore
	}
	return s.dataStores[name], nil
}

// Optimize returns the initial resource of a job.
func (s *BrainServer) Optimize(ctx context.Context, in *pb.OptimizeRequest) (*pb.OptimizeResponse, error) {
	return nil, nil
}

// GetJobMetrics returns a job metrics
func (s *BrainServer) GetJobMetrics(ctx context.Context, in *pb.JobMetricsRequest) (*pb.JobMetricsResponse, error) {
	dataStore, err := s.getDataStore(defaultDataStoreName)
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
			JobUUID: in.JobUuid,
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

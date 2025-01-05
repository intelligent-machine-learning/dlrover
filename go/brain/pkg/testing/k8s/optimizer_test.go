// Copyright 2023 The DLRover Authors. All rights reserved.
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

package k8s

import (
	"context"
	log "github.com/golang/glog"
	pb "github.com/intelligent-machine-learning/easydl/brain/pkg/proto"
	"github.com/intelligent-machine-learning/easydl/brain/pkg/utils"
	"google.golang.org/grpc"
	"testing"
	"time"
)

func TestOptimizer(t *testing.T) {
	if !toRunOptimizerTest {
		return
	}

	serverAddr := "brain.dlrover.svc.aliyun:50001"
	grpcTimeout := int32(5)

	conn, err := utils.NewRPCConnection(context.Background(), serverAddr, grpc.WithInsecure())
	if err != nil {
		log.Errorf("fail to create connection for %s: %v", serverAddr, err)
		return
	}
	ctx, cancel := context.WithTimeout(context.Background(), time.Duration(grpcTimeout)*time.Second)
	defer cancel()

	client := pb.NewBrainClient(conn)
	request := &pb.OptimizeRequest{
		Type: "",
		Config: &pb.OptimizeConfig{
			OptimizerConfigRetriever: "base_config_retriever",
			BrainProcessor:           "base_optimize_processor",
			CustomizedConfig: map[string]string{
				"optimizer": "base_optimizer",
			},
		},
	}
	resp, err := client.Optimize(ctx, request)
	if err != nil || !resp.Response.Success {
		log.Errorf("fail to optimize!!!!!")
	}
}

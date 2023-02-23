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

package utils

import (
	"context"
	log "github.com/golang/glog"
	"google.golang.org/grpc"
)

func NewRPCConnection(ctx context.Context, target string, opts ...grpc.DialOption) (conn *grpc.ClientConn, err error) {
	conn, err = grpc.DialContext(ctx, target, opts...)
	if err != nil {
		if conn != nil {
			conn.Close()
		}
		log.Errorf("Failed to connect server %s with err: %v ", target, err)
		return nil, err
	}
	return conn, err
}

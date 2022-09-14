#!/bin/bash

set -e

GO_MIRROR_URL=$1

curl --silent "$GO_MIRROR_URL"/go1.19.1.linux-amd64.tar.gz | \
    tar -C /usr/local -xzf -

go env -w GO111MODULE=on
go env -w GOPROXY=https://goproxy.io,direct

go install github.com/golang/protobuf/protoc-gen-go@latest > /dev/null
go install golang.org/x/lint/golint@latest > /dev/null
go install golang.org/x/tools/cmd/goyacc@latest > /dev/null
go install golang.org/x/tools/cmd/cover@latest > /dev/null
go install github.com/mattn/goveralls@latest > /dev/null
go install github.com/rakyll/gotest@latest > /dev/null
go install sigs.k8s.io/controller-runtime/tools/setup-envtest@latest

cp "$GOPATH"/bin/* /usr/local/bin/

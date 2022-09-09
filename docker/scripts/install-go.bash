#!/bin/bash

set -e

GO_MIRROR_URL=$1

curl --silent "$GO_MIRROR_URL"/go1.13.4.linux-amd64.tar.gz | \
    tar -C /usr/local -xzf -

go env -w GO111MODULE=on
go env -w GOPROXY=https://goproxy.io,direct

go get github.com/golang/protobuf/protoc-gen-go@v1.3.2 > /dev/null
go get golang.org/x/lint/golint > /dev/null
go get golang.org/x/tools/cmd/goyacc > /dev/null
go get golang.org/x/tools/cmd/cover > /dev/null
go get github.com/mattn/goveralls > /dev/null
go get github.com/rakyll/gotest > /dev/null

cp "$GOPATH"/bin/* /usr/local/bin/

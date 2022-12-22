FROM golang:1.18.9-bullseye as builder

# Copy in the go src
WORKDIR /go/src/github.com/intelligent-machine-learning/easydl/brain
COPY vendor/ vendor/
COPY cmd/brain cmd/brain
COPY cmd/k8smonitor cmd/k8smonitor
COPY pkg pkg
COPY go.mod go.mod

# Build
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -a -o brain \
    github.com/intelligent-machine-learning/easydl/brain/cmd/brain

RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -a -o k8smonitor \
    github.com/intelligent-machine-learning/easydl/brain/cmd/k8smonitor

# Copy file into a thin image
FROM easydl/easydl:7u2-common
WORKDIR /root/

COPY --from=builder /go/src/github.com/intelligent-machine-learning/easydl/brain/brain .
COPY --from=builder /go/src/github.com/intelligent-machine-learning/easydl/brain/k8smonitor .

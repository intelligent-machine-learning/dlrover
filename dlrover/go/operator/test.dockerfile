FROM golang:1.18 as base
WORKDIR /
COPY  manager .
USER 65532:65532

ENTRYPOINT ["/manager"]
#!/usr/bin/env bash

set -o errexit
set -o nounset
set -o pipefail
 
MODULE=github.com/intelligent-machine-learning/easydl/dlrover/go/operator

SCRIPT_ROOT=$(dirname "${BASH_SOURCE[0]}")/..
CODEGEN_PKG=${CODEGEN_PKG:-$(cd "${SCRIPT_ROOT}"; ls -d -1 ./vendor/k8s.io/code-generator 2>/dev/null || echo ../code-generator)}
 
source "${CODEGEN_PKG}/kube_codegen.sh"
 
kube::codegen::gen_client api \
	--output-dir pkg/client \
	--output-pkg ${MODULE}/pkg/client \
	--with-watch \
    --boilerplate "${SCRIPT_ROOT}/hack/boilerplate.go.txt"
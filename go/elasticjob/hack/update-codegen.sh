#!/usr/bin/env bash

set -o errexit
set -o nounset
set -o pipefail

# 1. Project Configuration
MODULE="github.com/intelligent-machine-learning/dlrover/go/elasticjob"
# Dynamically locate the code-generator package in Go module cache
CODEGEN_PKG=$(go list -m -f '{{.Dir}}' k8s.io/code-generator@v0.24.2 2>/dev/null || echo "$(go env GOPATH)/pkg/mod/k8s.io/code-generator@v0.24.2")

if [ ! -d "$CODEGEN_PKG" ]; then
    echo "❌ Error: k8s.io/code-generator@v0.24.2 not found."
    echo "Please run 'go get k8s.io/code-generator@v0.24.2' to install the dependency."
    exit 1
fi

# 2. Input/Output Paths
# Format: <group-name>/<api-version>:<full-package-path>
INPUT_PACKAGE="elastic.iml.github.io/v1alpha1:github.com/intelligent-machine-learning/dlrover/go/elasticjob/api/v1alpha1"
OUTPUT_PKG="${MODULE}/client"

ROOT_DIR=$(cd $(dirname "${BASH_SOURCE[0]}")/.. && pwd)
BOILERPLATE="${ROOT_DIR}/hack/boilerplate.go.txt"

# 3. Create Temporary Workspace for Codegen
CODEGEN_TMP=$(mktemp -d)
trap 'rm -rf "$CODEGEN_TMP"' EXIT

export GOPATH="$CODEGEN_TMP"

# Establish the spoofed path to trick client-gen into finding the API definitions
BASE_DIR="$CODEGEN_TMP/src/github.com/intelligent-machine-learning/dlrover/go/elasticjob"
mkdir -p "$BASE_DIR/pkg/apis/elastic"
mkdir -p "$BASE_DIR/api"

# Symlink actual API files to the spoofed directory structure
ln -s "$ROOT_DIR/api/v1alpha1" "$BASE_DIR/api/v1alpha1"
ln -s "$ROOT_DIR/api/v1alpha1" "$BASE_DIR/pkg/apis/elastic/v1alpha1"

echo "Step 1: Cleaning up old generated files..."
rm -rf "${ROOT_DIR}/client"

echo "Step 2: Generating DeepCopy methods..."
deepcopy-gen \
  --input-dirs $MODULE/api/v1alpha1 \
  -O zz_generated.deepcopy \
  --go-header-file "${BOILERPLATE}" \
  --output-base "$CODEGEN_TMP/src"

echo "Step 3: Generating Clientset (versioned)..."
client-gen \
  --clientset-name versioned \
  --input-base "" \
  --input "$MODULE/pkg/apis/elastic/v1alpha1" \
  --output-package "$MODULE/client/clientset" \
  --go-header-file "${BOILERPLATE}" \
  --output-base "$CODEGEN_TMP/src" \
  -v 5

echo "Step 4: Generating Listers..."
lister-gen \
  --input-dirs "$MODULE/api/v1alpha1" \
  --output-package "$MODULE/client/listers" \
  --go-header-file "${BOILERPLATE}" \
  --output-base "$CODEGEN_TMP/src" \
  -v 5

echo "Step 5: Generating Informers..."
informer-gen \
  --input-dirs "$MODULE/api/v1alpha1" \
  --versioned-clientset-package "$MODULE/client/clientset/versioned" \
  --listers-package "$MODULE/client/listers" \
  --output-package "$MODULE/client/informers" \
  --go-header-file "${BOILERPLATE}" \
  --output-base "$CODEGEN_TMP/src" \
  -v 5

# Move generated code back to the project root
mv "${CODEGEN_TMP}/src/${MODULE}/client/" "${ROOT_DIR}/"

echo "✅ Client code generation completed successfully!"

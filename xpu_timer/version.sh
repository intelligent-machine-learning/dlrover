#!/bin/bash
GIT_COMMIT=$(git rev-parse HEAD)
echo "STABLE_GIT_COMMIT $GIT_COMMIT"
echo "STABLE_BUILD_TIME \"$(date '+%Y-%m-%d %H:%M:%S')\""
echo "STABLE_BUILD_TYPE $1"
echo "STABLE_BUILD_PLATFORM $(cat .build_platform)"
echo "STABLE_BUILD_PLATFORM_VERSION $(cat .platform_version)"

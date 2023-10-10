// Modifications Copyright 2023 AntGroups, Inc.
// ATorch Team

// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#pragma once

#ifdef __HIP_PLATFORM_HCC__
#include <hip/hip_cooperative_groups.h>
#else
#include <cooperative_groups.h>
#endif

#include <cuda.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>

#include <cassert>
#include <iostream>

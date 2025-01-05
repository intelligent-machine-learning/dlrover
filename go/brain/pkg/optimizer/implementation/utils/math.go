// Copyright 2022 The DLRover Authors. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package utils

import (
	"sort"
)

// GetMaxJobNodeResource returns max resources of job nodes
func GetMaxJobNodeResource(resources map[uint64]float64) float64 {
	if resources == nil {
		return 0.0
	}
	maxRes := 0.0
	for _, res := range resources {
		if maxRes < res {
			maxRes = res
		}
	}
	return maxRes
}

// ComputeMajorCluster computes a cluster where the number of samples is bigger than 50%
func ComputeMajorCluster(nums []float64) []float64 {
	clusterValues := make([]float64, 0)
	if len(nums) == 0 {
		return clusterValues
	}
	sort.Float64s(nums)
	mediumIndex := len(nums) / 2
	clusterValues = append(clusterValues, nums[mediumIndex])
	leftIndex := mediumIndex - 1
	rightIndex := mediumIndex + 1
	for leftIndex >= 0 && rightIndex < len(nums) && len(clusterValues) < mediumIndex+1 {
		clusterKernel := clusterValues[len(clusterValues)/2]
		if clusterKernel-nums[leftIndex] < nums[rightIndex]-clusterKernel {
			clusterValues = append([]float64{nums[leftIndex]}, clusterValues...)
			leftIndex--
		} else {
			clusterValues = append(clusterValues, nums[rightIndex])
			rightIndex++
		}
	}
	return clusterValues
}

// ComputeAverage computes the average value of a float array
func ComputeAverage(nums []float64) float64 {
	sum := 0.0
	for i := 0; i < len(nums); i++ {
		sum += nums[i]
	}
	return sum / float64(len(nums))
}

// GetMaxUtil get the maximum resource utilization
func GetMaxUtil(useds map[uint64]float64, requireds map[uint64]float64) float64 {
	maxUtil := 0.0
	for n, used := range useds {
		required, found := requireds[n]
		if !found {
			continue
		}
		util := used / required

		if maxUtil == 0 || maxUtil < util {
			maxUtil = util
		}
	}
	return maxUtil
}

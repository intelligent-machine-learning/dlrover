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
	"github.com/intelligent-machine-learning/easydl/brain/pkg/common"
	"github.com/intelligent-machine-learning/easydl/brain/pkg/datastore/recorder/mysql"
	pb "github.com/intelligent-machine-learning/easydl/brain/pkg/proto"
)

// ConvertPBPodStateToPodState converts pb pod state to pod state
func ConvertPBPodStateToPodState(pbPod *pb.PodState) *common.PodState {
	if pbPod == nil {
		return nil
	}

	pod := &common.PodState{
		Name:  pbPod.Name,
		UUID:  pbPod.Uid,
		Type:  pbPod.Type,
		IsOOM: pbPod.IsOom,
	}

	if pbPod.CustomizedData != nil {
		pod.CustomizedData = make(map[string]string)
		for k, v := range pbPod.CustomizedData {
			pod.CustomizedData[k] = v
		}
	}
	return pod
}

// ConvertPodStateToPBPodState converts PodState to pb PodState
func ConvertPodStateToPBPodState(state *common.PodState) *pb.PodState {
	if state == nil {
		return nil
	}
	pbState := &pb.PodState{
		Name:  state.Name,
		Uid:   state.UUID,
		Type:  state.Type,
		IsOom: state.IsOOM,
	}

	if state.CustomizedData != nil {
		pbState.CustomizedData = make(map[string]string)
		for k, v := range state.CustomizedData {
			pbState.CustomizedData[k] = v
		}
	}
	return pbState
}

// ConvertPBJobStateToJobState converts pb job state to job state
func ConvertPBJobStateToJobState(pbJob *pb.JobState) *common.JobState {
	if pbJob == nil {
		return nil
	}
	job := &common.JobState{
		PodStates: make(map[string]*common.PodState),
	}

	if pbJob.Pods != nil {
		for name, pod := range pbJob.Pods {
			job.PodStates[name] = ConvertPBPodStateToPodState(pod)
		}
	}

	if pbJob.CustomizedData != nil {
		job.CustomizedData = make(map[string]string)
		for k, v := range pbJob.CustomizedData {
			job.CustomizedData[k] = v
		}
	}
	return job
}

// ConvertJobStateToPBJobState converts JobState to pb JobState
func ConvertJobStateToPBJobState(state *common.JobState) *pb.JobState {
	if state == nil {
		return nil
	}
	pbState := &pb.JobState{}
	if state.PodStates != nil {
		pbState.Pods = make(map[string]*pb.PodState)
		for pod, podState := range state.PodStates {
			pbState.Pods[pod] = ConvertPodStateToPBPodState(podState)
		}
	}

	if state.CustomizedData != nil {
		pbState.CustomizedData = make(map[string]string)
		for k, v := range state.CustomizedData {
			pbState.CustomizedData[k] = v
		}
	}
	return pbState
}

// ConvertPBOptimizeJobMetaToJobMeta converts pb OptimizeJobMeta to JobMeta
func ConvertPBOptimizeJobMetaToJobMeta(job *pb.OptimizeJobMeta) *common.JobMeta {
	if job == nil {
		return nil
	}
	return &common.JobMeta{
		UUID:      job.Uid,
		Cluster:   job.Cluster,
		Namespace: job.Namespace,
		State:     ConvertPBJobStateToJobState(job.State),
	}
}

// ConvertJobMetaToPBOptimizeJobMeta converts JobMeta to pb OptimizeJobMeta
func ConvertJobMetaToPBOptimizeJobMeta(job *common.JobMeta) *pb.OptimizeJobMeta {
	if job == nil {
		return nil
	}
	return &pb.OptimizeJobMeta{
		Uid:       job.UUID,
		Cluster:   job.Cluster,
		Namespace: job.Namespace,
		State:     ConvertJobStateToPBJobState(job.State),
	}
}

// ConvertPBJobMetaToJobMeta converts pb JobMeta to JobMeta
func ConvertPBJobMetaToJobMeta(job *pb.JobMeta) *common.JobMeta {
	if job == nil {
		return nil
	}
	return &common.JobMeta{
		UUID: job.Uuid,
		Name: job.Name,
		User: job.User,
	}
}

// ConvertPBJobMetaArrayToJobMetaArray converts pb JobMeta array to JobMeta array
func ConvertPBJobMetaArrayToJobMetaArray(pbJobs []*pb.JobMeta) []*common.JobMeta {
	if pbJobs == nil {
		return nil
	}

	jobs := make([]*common.JobMeta, 0)
	for _, pbJob := range pbJobs {
		jobs = append(jobs, ConvertPBJobMetaToJobMeta(pbJob))
	}
	return jobs
}

// ConvertPodResourceToPBPodResource converts PodResource to pb PodResource
func ConvertPodResourceToPBPodResource(res *common.PodResource) *pb.PodResource {
	if res == nil {
		return nil
	}
	return &pb.PodResource{
		Memory:  int64(res.Memory),
		Cpu:     res.CPUCore,
		Gpu:     res.GPUCore,
		GpuType: res.GPUType,
	}
}

// ConvertPBPodResourceToPodResource converts pb PodResource to PodResource
func ConvertPBPodResourceToPodResource(res *pb.PodResource) *common.PodResource {
	if res == nil {
		return nil
	}
	return &common.PodResource{
		Memory:  float64(res.Memory),
		CPUCore: res.Cpu,
		GPUCore: res.Gpu,
		GPUType: res.GpuType,
	}
}

// ConvertTaskGroupResourceToPBTaskGroupResource converts TaskGroupResource to pb TaskGroupResource
func ConvertTaskGroupResourceToPBTaskGroupResource(res *common.TaskGroupResource) *pb.TaskGroupResource {
	if res == nil {
		return nil
	}
	return &pb.TaskGroupResource{
		Count:    int64(res.Count),
		Resource: ConvertPodResourceToPBPodResource(res.Resource),
	}
}

// ConvertPBTaskGroupResourceToTaskGroupResource converts pb TaskGroupResource to TaskGroupResource
func ConvertPBTaskGroupResourceToTaskGroupResource(res *pb.TaskGroupResource) *common.TaskGroupResource {
	if res == nil {
		return nil
	}
	return &common.TaskGroupResource{
		Count:    int32(res.Count),
		Resource: ConvertPBPodResourceToPodResource(res.Resource),
	}
}

// ConvertJobResourceToPBJobResource converts JobResource to pb JobResource
func ConvertJobResourceToPBJobResource(jobRes *common.JobResource) *pb.JobResource {
	if jobRes == nil {
		return nil
	}
	podRes := make(map[string]*pb.PodResource)
	for name, res := range jobRes.PodResources {
		podRes[name] = ConvertPodResourceToPBPodResource(res)
	}

	taskGroupRes := make(map[string]*pb.TaskGroupResource)
	for name, res := range jobRes.TaskGroupResources {
		taskGroupRes[name] = ConvertTaskGroupResourceToPBTaskGroupResource(res)
	}

	return &pb.JobResource{
		PodResources:       podRes,
		TaskGroupResources: taskGroupRes,
	}
}

// ConvertPBJobResourceToJobResource converts pb JobResource to JobResource
func ConvertPBJobResourceToJobResource(jobRes *pb.JobResource) *common.JobResource {
	if jobRes == nil {
		return nil
	}
	podRes := make(map[string]*common.PodResource)
	for name, res := range jobRes.PodResources {
		podRes[name] = ConvertPBPodResourceToPodResource(res)
	}

	taskGroupRes := make(map[string]*common.TaskGroupResource)
	for name, res := range jobRes.TaskGroupResources {
		taskGroupRes[name] = ConvertPBTaskGroupResourceToTaskGroupResource(res)
	}

	return &common.JobResource{
		PodResources:       podRes,
		TaskGroupResources: taskGroupRes,
	}
}

// ConvertOptimizePlanToPBJobOptimizePlan converts OptimizePlan to pb JobOptimizePlan
func ConvertOptimizePlanToPBJobOptimizePlan(plan *common.OptimizePlan) *pb.JobOptimizePlan {
	if plan == nil {
		return nil
	}
	return &pb.JobOptimizePlan{
		Job:      ConvertJobMetaToPBOptimizeJobMeta(plan.JobMeta),
		Resource: ConvertJobResourceToPBJobResource(plan.AlgOptPlan.JobRes),
	}
}

// ConvertPBJobOptimizePlanToOptimizePlan converts pb JobOptimizePlan to OptimizePlan
func ConvertPBJobOptimizePlanToOptimizePlan(plan *pb.JobOptimizePlan) *common.OptimizePlan {
	if plan == nil {
		return nil
	}
	return &common.OptimizePlan{
		JobMeta: ConvertPBOptimizeJobMetaToJobMeta(plan.Job),
		AlgOptPlan: &common.AlgorithmOptimizePlan{
			JobRes: ConvertPBJobResourceToJobResource(plan.Resource),
		},
	}
}

// ConvertDBJobMetricsToJobMetrics converts db JobMetrics to JobMetrics
func ConvertDBJobMetricsToJobMetrics(job *mysql.JobMetrics) *common.JobMetrics {
	if job == nil {
		return nil
	}
	return &common.JobMetrics{
		JobUUID:            job.UID,
		HyperParamsFeature: job.HyperParamsFeature,
		JobFeature:         job.JobFeature,
		DatasetFeature:     job.DatasetFeature,
		ModelFeature:       job.ModelFeature,
		JobRuntime:         job.JobRuntime,
		ExitReason:         job.ExitReason,
		Optimization:       job.Optimization,
		Type:               job.Type,
		Resource:           job.Resource,
		CustomizedData:     job.CustomizedData,
	}
}

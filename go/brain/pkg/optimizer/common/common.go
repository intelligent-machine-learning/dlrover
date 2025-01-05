// Copyright 2023 The DLRover Authors. All rights reserved.
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

package common

const (
	// JobStageCreate indicate job is at create stage
	JobStageCreate = "job_stage_create"
	// JobStagePSInitial indicate job is at ps initial stage
	JobStagePSInitial = "job_stage_ps_initial"
	// JobStageWorkerInitial indicate job is at worker initial stage
	JobStageWorkerInitial = "job_stage_worker_initial"
	// JobStageRunning indicate job is at running stage
	JobStageRunning = "job_stage_running"
)

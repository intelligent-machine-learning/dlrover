package optimizer

import (
	"encoding/json"
	"github.com/intelligent-machine-learning/easydl/brain/pkg/common"
	"github.com/intelligent-machine-learning/easydl/brain/pkg/config"
	dsimpl "github.com/intelligent-machine-learning/easydl/brain/pkg/datastore/implementation"
	"github.com/intelligent-machine-learning/easydl/brain/pkg/datastore/recorder/mysql"
	optconfig "github.com/intelligent-machine-learning/easydl/brain/pkg/optimizer/config"
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestJobPSRunningResourceOptimizer(t *testing.T) {
	jobUUID := "uuid"

	dataStore := &dsimpl.BaseDataStore{
		Client: mysql.NewFakeClient(),
	}

	///////////////////////////////////////////////////////////
	cpu1 := 6
	cpu2 := 9
	memory := 4 * 1024 * 1024 * 1024
	runtimes := []*common.JobRuntimeInfo{
		{
			PSCPU: map[uint64]float64{
				0: float64(cpu1),
				1: float64(cpu2),
			},
			PSMemory: map[uint64]float64{
				0: float64(memory),
				1: float64(memory),
			},
		},
		{
			PSCPU: map[uint64]float64{
				0: float64(cpu1),
				1: float64(cpu2),
			},
			PSMemory: map[uint64]float64{
				0: float64(memory),
				1: float64(memory),
			},
		},
		{
			PSCPU: map[uint64]float64{
				0: float64(cpu1),
				1: float64(cpu2),
			},
			PSMemory: map[uint64]float64{
				0: float64(memory),
				1: float64(memory),
			},
		},
	}
	runtimeStr, err := json.Marshal(runtimes)
	assert.NoError(t, err)

	jobMetrics := &mysql.JobMetrics{
		JobUUID:    jobUUID,
		JobRuntime: string(runtimeStr),
	}
	dataStore.Client.JobMetricsRecorder.Upsert(jobMetrics)

	////////////////////////////////////////////////////////////////
	nodeName1 := "task-ps-0"
	nodeName2 := "task-ps-1"

	nodeRes := &common.PodResource{
		CPUCore: 10,
	}
	nodeResStr, _ := json.Marshal(nodeRes)

	node1 := &mysql.JobNode{
		Name:     nodeName1,
		UID:      "task-uid-1",
		JobUUID:  jobUUID,
		Type:     common.PSTaskGroupName,
		Resource: string(nodeResStr),
	}
	node2 := &mysql.JobNode{
		Name:     nodeName2,
		UID:      "task-uid-2",
		JobUUID:  jobUUID,
		Type:     common.PSTaskGroupName,
		Resource: string(nodeResStr),
	}
	dataStore.Client.JobNodeRecorder.Upsert(node1)
	dataStore.Client.JobNodeRecorder.Upsert(node2)

	////////////////////////////////////////////////////////////////////
	jobMetas := []*common.JobMeta{
		{
			Name: "job",
			UUID: jobUUID,
		},
	}

	optimizerConf := &optconfig.OptimizerConfig{
		OptimizeAlgorithmConfig: &optconfig.OptimizeAlgorithmConfig{},
	}

	conf := config.NewEmptyConfig()
	testOptimizer := newJobPSRunningResourceOptimizer(dataStore, conf)
	plans, err := testOptimizer.Optimize(optimizerConf, jobMetas)

	assert.NoError(t, err)
	assert.Equal(t, len(plans), 1)
	res := plans[0].AlgOptPlan.JobRes.PodResources[nodeName2]
	assert.Equal(t, res.CPUCore, float32(32))
}
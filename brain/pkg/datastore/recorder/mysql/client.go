package mysql

import "github.com/intelligent-machine-learning/easydl/brain/pkg/datastore/dbbase"

// Client is the struct of mysql db client
type Client struct {
	JobMetricsRecorder JobMetricsRecorderInterface
	TaskGroupsRecorder TaskGroupsRecorderInterface
	TasksRecorder      TasksRecorderInterface
}

// NewClient returns a new mysql db client
func NewClient(db *dbbase.DB) *Client {
	return &Client{
		JobMetricsRecorder: NewJobMetricsDBRecorder(db),
		TaskGroupsRecorder: NewTaskGroupsDBRecorder(db),
		TasksRecorder:      NewTasksRecorder(db),
	}
}

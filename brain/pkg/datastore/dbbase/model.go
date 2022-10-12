package dbbase

import "time"

// TimeRange presents time range
type TimeRange struct {
	// From (inclusive, epoch time)
	From time.Time
	// To (inclusive, epoch time)
	To time.Time
}

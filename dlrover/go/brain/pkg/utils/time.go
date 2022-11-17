package utils

import "time"

// Condition is the condition function
type Condition func() bool

// WaitForCondition waits until meets the condtion
func WaitForCondition(cond Condition, checkInterval time.Duration, timeout time.Duration) bool {
	tick := time.NewTicker(checkInterval)
	expire := time.NewTimer(timeout)

	defer func() {
		tick.Stop()
		expire.Stop()
	}()

	for {
		select {
		case <-tick.C:
			if done := cond(); done {
				return true
			}
		case <-expire.C:
			return false
		}
	}
}

const (
	secondThreshold = int64(100000000000)
)

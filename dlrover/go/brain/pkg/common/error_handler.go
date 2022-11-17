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

package common

import (
	"context"
	"errors"
	"fmt"
	log "github.com/golang/glog"
)

// Error is a structured error
type Error struct {
	// ComponentName indicates service name where the error occurs in.
	ComponentName string

	// RawError indicates the raw error.
	RawError error

	// ExtraInfo is extra information
	ExtraInfos map[string]interface{}
}

// ErrorReporter can report Error to a ErrorHandler.
type ErrorReporter interface {

	// ReportError reports error to ErrorHandler. This method may invoke a blocked IO operation,
	// so the invoker should send it's context to cancel the underlying blocked io operation if invoker routine is stopped.
	ReportError(ctx context.Context, err Error)
}

// ErrorHandler handles Error reported by ErrorReporter.
type ErrorHandler interface {
	// We can also send error to myself.
	ErrorReporter

	// HandleError handles reported error.
	HandleError(ctx context.Context)
}

// NewStringError returns a Error.
func NewStringError(componentName, err string) Error {
	return Error{
		ComponentName: componentName,
		RawError:      errors.New(err),
	}
}

// NewError returns a Error.
func NewError(componentName string, err error) Error {
	return Error{
		ComponentName: componentName,
		RawError:      err,
	}
}

// NewStopAllErrorHandler returns a stopAllErrorHandler. Returns an error if input cancelFunc is nil.
func NewStopAllErrorHandler(cancelFunc context.CancelFunc) (ErrorHandler, error) {
	if cancelFunc == nil {
		return nil, fmt.Errorf("cancelFunc is empty")
	}

	return &stopAllErrorHandler{
		ch:         make(chan Error, 1),
		cancelFunc: cancelFunc,
	}, nil
}

// stopAllErrorHandler implements ErrorHandler. It would stop the root context if receive an error.
type stopAllErrorHandler struct {
	ch         chan Error
	cancelFunc context.CancelFunc
}

// ReportError reports input Error to underlying channel
func (s *stopAllErrorHandler) ReportError(ctx context.Context, err Error) {
	select {
	case <-ctx.Done():
		return
	case s.ch <- err:
	}
}

// HandleError stops the context if receive an error.
func (s *stopAllErrorHandler) HandleError(ctx context.Context) {
	select {
	case <-ctx.Done():
		log.Infof("context is %v, exit error check", ctx.Err())
	case e := <-s.ch:
		log.Infof("Receive error: %v", e)
		s.cancelFunc()
	}
}

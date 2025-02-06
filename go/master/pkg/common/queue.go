// Copyright 2025 The DLRover Authors. All rights reserved.
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
	"container/list"
	"sync"
)

// Queue is a thread-safe queue
type Queue struct {
	lock sync.Mutex
	data *list.List
}

// NewQueue creates a Queue instance.
func NewQueue() *Queue {
	q := new(Queue)
	q.data = list.New()
	q.lock = sync.Mutex{}
	return q
}

// PushFront pushes an element at the head of the queue.
func (q *Queue) PushFront(v interface{}) {
	defer q.lock.Unlock()
	q.lock.Lock()
	q.data.PushFront(v)
}

// PushBack pushes an element at the back of the queue.
func (q *Queue) PushBack(v interface{}) {
	defer q.lock.Unlock()
	q.lock.Lock()
	q.data.PushBack(v)
}

// PopFront gets the front element and removes it from the queue.
func (q *Queue) PopFront() interface{} {
	defer q.lock.Unlock()
	q.lock.Lock()
	iter := q.data.Front()
	v := iter.Value
	q.data.Remove(iter)
	return v
}

// PopBack gets the back element and removes it from the queue.
func (q *Queue) PopBack() interface{} {
	defer q.lock.Unlock()
	q.lock.Lock()
	iter := q.data.Back()
	v := iter.Value
	q.data.Remove(iter)
	return v
}

// Len gets the number of elements in the queue.
func (q *Queue) Len() int {
	defer q.lock.Unlock()
	q.lock.Lock()
	return q.data.Len()
}

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

package config

import (
	"context"
	"fmt"
	"github.com/stretchr/testify/assert"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes/fake"
	"testing"
	"time"
)

// TestController tests config controller
func TestController(t *testing.T) {
	var err error
	var k8sCM, getCM, createCM *v1.ConfigMap

	k8sClientSet := fake.NewSimpleClientset()
	namespace := "test-namespace"
	name := fmt.Sprintf("test-getCM-%d", time.Now().Unix())
	k8sCM = &v1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: namespace,
			Name:      name,
		},
		Data: map[string]string{
			"key1": "value1",
		},
	}

	ctx, cancelFunc := context.WithCancel(context.Background())
	defer cancelFunc()

	confController := NewController(namespace, name, k8sClientSet)

	// Get before ConfigMap created
	getCM, err = confController.Get(ctx)
	assert.NotEqual(t, err, nil)
	assert.Equal(t, getCM, (*v1.ConfigMap)(nil))

	opt := metav1.CreateOptions{}
	// Create ConfigMap
	createCM, err = k8sClientSet.CoreV1().ConfigMaps(namespace).Create(ctx, k8sCM, opt)
	assert.Equal(t, err, nil)
	assert.NotEqual(t, createCM, nil)

	// Get again
	getCM, err = confController.Get(ctx)
	assert.Equal(t, err, nil)
	assert.NotEqual(t, createCM, nil)
}

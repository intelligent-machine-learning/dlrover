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
	"fmt"
	log "github.com/golang/glog"
	"gopkg.in/yaml.v3"
	"k8s.io/client-go/kubernetes"
)

// Config is the struct of the config
type Config struct {
	data map[string]interface{}
}

// NewEmptyConfig returns a new empty config
func NewEmptyConfig() *Config {
	return &Config{
		data: make(map[string]interface{}),
	}
}

// NewConfig Return a config instance with the specified config.
func NewConfig(conf map[string]interface{}) *Config {
	config := NewEmptyConfig()
	config.SetData(conf)
	return config
}

// SetConfig sets the config
func (c *Config) SetConfig(config *Config) {
	c.data = make(map[string]interface{})
	for key, val := range config.data {
		c.data[key] = val
	}
}

// SetData sets the config data
func (c *Config) SetData(data map[string]interface{}) error {
	c.data = make(map[string]interface{})
	if data == nil {
		err := fmt.Errorf("[Config] data is empty")
		return err
	}
	for key, val := range data {
		c.data[key] = val
	}
	return nil
}

// Clone returns a copy of the config
func (c *Config) Clone() *Config {
	conf := NewEmptyConfig()
	conf.SetConfig(c)
	return conf
}

// Contains check if contains a key-value
func (c *Config) Contains(key string) bool {
	_, exist := c.data[key]
	return exist
}

// IsEmpty check if the config is empty
func (c *Config) IsEmpty() bool {
	return len(c.data) == 0
}

// GetString returns a string value for a given key
func (c *Config) GetString(key string) string {
	val, exists := c.data[key]
	if !exists {
		log.Errorf("key %s does not exist in %v", key, c)
		return ""
	}
	return val.(string)
}

// GetInt returns a int value for a given key
func (c *Config) GetInt(key string) int {
	_, exist := c.data[key]
	if !exist {
		return 0
	}
	return c.data[key].(int)
}

// GetStringArray returns string array
func (c *Config) GetStringArray(key string) []string {
	rawValue, exist := c.data[key]
	if !exist {
		return nil
	}

	value := make([]string, 0)

	switch rawValue.(type) {
	case []string:
		for _, v := range rawValue.([]string) {
			value = append(value, v)
		}
		break
	case []interface{}:
		for _, v := range rawValue.([]interface{}) {
			value = append(value, v.(string))
		}
		break
	}
	return value
}

// GetBool returns a bool value for a given key
func (c *Config) GetBool(key string) bool {
	_, exist := c.data[key]
	if !exist {
		return false
	}
	return c.data[key].(bool)
}

// GetIntWithValue returns a int value for a given key, if not found, return default value
func (c *Config) GetIntWithValue(key string, defaultVal int) int {
	_, exist := c.data[key]
	if !exist {
		return defaultVal
	}
	return c.data[key].(int)
}

// GetFloat64 returns a float value for a given key
func (c *Config) GetFloat64(key string) float64 {
	_, exist := c.data[key]
	if !exist {
		return 0.0
	}
	return c.data[key].(float64)
}

// GetFloat64WithValue returns a float value for a given key, if not found, return default value
func (c *Config) GetFloat64WithValue(key string, defaultVal float64) float64 {
	_, exist := c.data[key]
	if !exist {
		return defaultVal
	}
	return c.data[key].(float64)
}

// Set sets a key-value pair
func (c *Config) Set(key string, value interface{}) {
	c.data[key] = value
}

// GetKubeClientInterface returns kube interface
func (c *Config) GetKubeClientInterface() kubernetes.Interface {
	if _, exists := c.data[KubeClientInterface]; !exists {
		return nil
	}
	return c.data[KubeClientInterface].(kubernetes.Interface)
}

// GetConfig returns the config for a given key
func (c *Config) GetConfig(key string) *Config {
	value, exist := c.data[key]
	if !exist {
		return nil
	}

	serialized, err := yaml.Marshal(value)
	if err != nil {
		log.Warningf("failed to serialize conf [%v], %v", value, err)
		return nil
	}
	mapVal := make(map[string]interface{})
	if err := yaml.Unmarshal(serialized, mapVal); err != nil {
		log.Warningf("failed to deserialize [%v] to Map, %v", string(serialized), err)
		return nil
	}
	return NewConfig(mapVal)
}

// GetKeys returns all keys in the config
func (c *Config) GetKeys() []string {
	keys := make([]string, 0)
	for key := range c.data {
		keys = append(keys, key)
	}
	return keys
}

// Get returns the value for a given key
func (c *Config) Get(key string) interface{} {
	value, exists := c.data[key]
	if !exists {
		return nil
	}
	return value
}

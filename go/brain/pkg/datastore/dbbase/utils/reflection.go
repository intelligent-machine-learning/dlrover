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
	"fmt"
	"github.com/elliotchance/orderedmap"
	"reflect"
)

// IsPtr check if obj is a pointer
func IsPtr(obj interface{}) bool {
	return reflect.ValueOf(obj).Kind() == reflect.Ptr
}

// ToMap converts a struct object or point to object to a map. We have to use ordered map to preserve field order for testing.
// Zero values are ignored.
func ToMap(obj interface{}) (*orderedmap.OrderedMap, error) {
	value := reflect.Indirect(reflect.ValueOf(obj))
	if value.Kind() != reflect.Struct {
		return nil, fmt.Errorf("Kind mismatch: can't convert %v to map", obj)
	}
	ret := orderedmap.NewOrderedMap()
	typ := value.Type()
	for i := 0; i < typ.NumField(); i++ {
		fieldName := typ.Field(i).Name
		fieldValue := value.Field(i)
		if !fieldValue.IsZero() {
			ret.Set(fieldName, value.Field(i).Interface())
		}
	}
	return ret, nil
}

// ToTagMap returns {fieldName: tagValue} pairs.
// If tagKey doesn't exist, this field will be ignored.
func ToTagMap(obj interface{}, tagKey string) (*orderedmap.OrderedMap, error) {
	value := reflect.Indirect(reflect.ValueOf(obj))
	if value.Kind() != reflect.Struct {
		return nil, fmt.Errorf("Kind mismatch: can't convert %v to map", obj)
	}
	ret := orderedmap.NewOrderedMap()
	typ := value.Type()
	for i := 0; i < typ.NumField(); i++ {
		field := typ.Field(i)
		if tagValue, ok := field.Tag.Lookup(tagKey); ok {
			ret.Set(field.Name, tagValue)
		}
	}
	return ret, nil
}

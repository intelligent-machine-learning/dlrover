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

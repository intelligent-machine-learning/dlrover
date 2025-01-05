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

package dbbase

import (
	"errors"
	"fmt"
	"github.com/elliotchance/orderedmap"
	log "github.com/golang/glog"
	"github.com/intelligent-machine-learning/easydl/brain/pkg/datastore/dbbase/utils"
	"reflect"
	"strings"
	"xorm.io/xorm"
)

var (
	xormTagMapCache = map[reflect.Type]*orderedmap.OrderedMap{}
	// Keyword: SkipNext
	skipKeywords = map[string]bool{
		"EXTENDS":  false,
		"<-":       false,
		"->":       false,
		"PK":       false,
		"NULL":     false,
		"NOT":      false,
		"AUTOINCR": false,
		"DEFAULT":  true,
		"CREATED":  false,
		"UPDATED":  false,
		"DELETED":  false,
		"VERSION":  false,
		"UTC":      false,
		"LOCAL":    false,
		"NOTNULL":  false,
		"INDEX":    false,
		"UNIQUE":   false,
		"CACHE":    false,
		"NOCACHE":  false,
		"COMMENT":  false,
	}
)

// Condition is the interface of db query condition
type Condition interface {
	// Apply this condition to SQL where clause
	Apply(session *xorm.Session) *xorm.Session
}

// RecorderInterface is the interface of the recorder
type RecorderInterface interface {
	// row must be a pointer
	Get(row interface{}, condition Condition) error
	// Rows must be a pointer to Rows
	List(rows interface{}, condition Condition) error
	Count(condition Condition) (uint64, error)
	Upsert(row interface{}) error
	UpsertMany(rows []interface{}) error
}

// DatabaseRecorder is the struct of the database recorder
type DatabaseRecorder struct {
	*xorm.Engine
	TableName string
}

var _ RecorderInterface = &DatabaseRecorder{}

// Get returns a single row which meets the condition
func (r *DatabaseRecorder) Get(row interface{}, condition Condition) error {
	if !utils.IsPtr(row) {
		return errors.New("should pass a pointer to 'Get'")
	}
	session := r.Table(r.TableName)
	session = condition.Apply(session)
	found, err := session.Get(row)
	if err != nil {
		log.Errorf("Failed to get %v of %+v: %v", r.TableName, condition, err)
		return err
	}
	if !found {
		err = fmt.Errorf("Can't find %v of %+v: %v", r.TableName, condition, err)
		return err
	}
	return nil
}

// List returns multiple rows which meet the condition
func (r *DatabaseRecorder) List(rows interface{}, condition Condition) error {
	if !utils.IsPtr(rows) {
		return errors.New("should pass a pointer to 'List'")
	}
	session := r.Table(r.TableName)
	session = condition.Apply(session)
	err := session.Find(rows)
	if err != nil {
		log.Errorf("Failed to list %v of %+v: %v", r.TableName, condition, err)
		return err
	}
	return err
}

// Count returns the number of rows which meet the condition
func (r *DatabaseRecorder) Count(condition Condition) (uint64, error) {
	session := r.Table(r.TableName)
	session = condition.Apply(session)
	count, err := session.Count()
	if err != nil {
		log.Errorf("Failed to count %v of %+v: %v", r.TableName, condition, err)
		return 0, err
	}
	return uint64(count), nil
}

// Upsert updates or insert a row
func (r *DatabaseRecorder) Upsert(row interface{}) error {
	session := r.Table(r.TableName)
	return r.upsertInner(session, row)
}

// UpsertMany updates or insert many rows
func (r *DatabaseRecorder) UpsertMany(rows []interface{}) error {
	_, err := r.Transaction(func(session *xorm.Session) (interface{}, error) {
		for _, row := range rows {
			if err := r.upsertInner(session, row); err != nil {
				return nil, err
			}
		}
		return nil, nil
	})
	return err
}

func (r *DatabaseRecorder) upsertInner(session *xorm.Session, row interface{}) error {
	if !utils.IsPtr(row) {
		return errors.New("should pass pointer to Upsert/UpsertMany")
	}
	rowMap, err := r.toDBMap(row)
	if err != nil {
		return err
	}
	if rowMap.Len() == 0 {
		// Nothing to update, skip this row
		return nil
	}
	sqlAndArgs := generateInsertOnDupUpdateSQLAndArgs(r.TableName, rowMap)
	if _, err = session.Exec(sqlAndArgs...); err != nil {
		return err
	}
	return nil
}

func generateInsertOnDupUpdateSQLAndArgs(tableName string, rowMap *orderedmap.OrderedMap) []interface{} {
	keys, args, placeholders, updatePlaceholders := make([]string, 0), make([]interface{}, 0), make([]string, 0), make([]string, 0)
	for el := rowMap.Front(); el != nil; el = el.Next() {
		key, arg := el.Key, el.Value
		keys = append(keys, fmt.Sprintf("`%v`", key))
		args = append(args, arg)
		placeholders = append(placeholders, "?")
		updatePlaceholders = append(updatePlaceholders, fmt.Sprintf("`%v` = ?", key))
	}
	sql := fmt.Sprintf("INSERT INTO `%v` (%v) VALUES (%v) ON DUPLICATE KEY UPDATE %s",
		tableName, strings.Join(keys, ", "),
		strings.Join(placeholders, ", "),
		strings.Join(updatePlaceholders, ", "))
	sqlAndArgs := []interface{}{sql}
	sqlAndArgs = append(sqlAndArgs, args...)
	sqlAndArgs = append(sqlAndArgs, args...)
	return sqlAndArgs
}

func (r *DatabaseRecorder) toDBMap(row interface{}) (*orderedmap.OrderedMap, error) {
	mapper := r.GetColumnMapper()
	structFieldNameMap, err := utils.ToMap(row)
	if err != nil {
		return nil, err
	}
	tagMap, err := ToXormTagMapCached(row)
	if err != nil {
		return nil, err
	}
	ret := orderedmap.NewOrderedMap()
	for el := structFieldNameMap.Front(); el != nil; el = el.Next() {
		structFieldName := el.Key.(string)
		dbFieldName := ""
		// Get DB field name from xorm tag if exists
		if tagName, ok := tagMap.Get(structFieldName); ok {
			dbFieldName = tagName.(string)
		} else {
			// Or from struct field name
			dbFieldName = mapper.Obj2Table(structFieldName)
		}
		ret.Set(dbFieldName, el.Value)
	}
	return ret, nil
}

// ToXormTagMapCached caches toXormTagMap results. It's ~100x faster than toXormTagMap.
// Cached will not work If toXormTagMap returns error.
func ToXormTagMapCached(obj interface{}) (*orderedmap.OrderedMap, error) {
	typ := reflect.Indirect(reflect.ValueOf(obj)).Type()
	// Number of xorm models is limited, plain map cache should be decent.
	if xormTagMap, ok := xormTagMapCache[typ]; ok {
		return xormTagMap, nil
	}
	xormTagMap, err := toXormTagMap(obj)
	if err != nil {
		return nil, err
	}
	xormTagMapCache[typ] = xormTagMap
	return xormTagMap, nil
}

// toXormTagMap returns {fieldName: xormTagFieldName} pairs. Field will be ignored if xorm tag doesn't exist.
func toXormTagMap(obj interface{}) (*orderedmap.OrderedMap, error) {
	tagMap, err := utils.ToTagMap(obj, "xorm")
	if err != nil {
		return nil, err
	}
	ret := orderedmap.NewOrderedMap()
	for el := tagMap.Front(); el != nil; el = el.Next() {
		tags := splitTag(el.Value.(string))
		if len(tags) == 0 || tags[0] == "-" {
			continue
		}
		skipNext := false
		tagName := ""
		for _, key := range tags {
			if skipNext {
				skipNext = false
				continue
			}
			if sn, ok := skipKeywords[strings.ToUpper(key)]; ok {
				skipNext = sn
				continue
			}
			if strings.HasPrefix(key, "'") && strings.HasSuffix(key, "'") {
				tagName = key[1 : len(key)-1]
				break
			} else {
				tagName = key
			}
		}
		if tagName != "" {
			ret.Set(el.Key, tagName)
		}
	}
	return ret, nil
}

// splitTag is copied from xorm to split xorm tag
func splitTag(tag string) (tags []string) {
	tag = strings.TrimSpace(tag)
	var hasQuote = false
	var lastIdx = 0
	for i, t := range tag {
		if t == '\'' {
			hasQuote = !hasQuote
		} else if t == ' ' {
			if lastIdx < i && !hasQuote {
				tags = append(tags, strings.TrimSpace(tag[lastIdx:i]))
				lastIdx = i + 1
			}
		}
	}
	if lastIdx < len(tag) {
		tags = append(tags, strings.TrimSpace(tag[lastIdx:]))
	}
	return
}

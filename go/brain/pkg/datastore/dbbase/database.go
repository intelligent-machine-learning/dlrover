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
	"database/sql"
	"fmt"
	"github.com/DATA-DOG/go-sqlmock"
	// mysql driver
	_ "github.com/go-sql-driver/mysql"
	log "github.com/golang/glog"
	"strings"
	"time"
	"xorm.io/xorm"
	"xorm.io/xorm/names"
)

// Database is the struct of database
type Database struct {
	*xorm.Engine
}

// NewDatabase creates a DB
func NewDatabase(username, password, engineType, url string) *Database {
	var db Database
	uri := formatURI(username, password, url)
	db.init(engineType, uri)
	return &db
}

// Make sure the database connection URL is well formatted
func formatURI(username, password, url string) string {
	var params []string
	if !strings.Contains(url, "interpolateParams=") {
		params = append(params, "interpolateParams=true")
	}
	if !strings.Contains(url, "parseTime=") {
		params = append(params, "parseTime=true")
	}
	if !strings.Contains(url, "clientFoundRows=") {
		params = append(params, "clientFoundRows=true")
	}
	if !strings.Contains(url, "charset=") {
		params = append(params, "charset=utf8mb4,utf8")
	}
	if len(params) > 0 {
		if !strings.Contains(url, "?") {
			url += "?"
		} else {
			url += "&"
		}
		url += strings.Join(params, "&")
	}
	log.Infof("Database URL is formatted as %s", url)
	uri := fmt.Sprintf("%s:%s@%s", username, password, url)
	return uri
}

func (db *Database) init(engineType string, uri string) {
	var err error
	engine, err := xorm.NewEngine(engineType, uri)
	if err != nil {
		panic(err)
	}
	// Test DB availability as early as possible
	if err = engine.Ping(); err != nil {
		panic(err)
	}
	db.Engine = postProcessEngine(engine, true)
}

// postProcessEngine set default mapper, fix time zone problem and set show sql
func postProcessEngine(engine *xorm.Engine, showSQL bool) *xorm.Engine {
	uri := engine.DataSourceName()
	// Set Gonic mapper, for example: WorkerGPU <==> worker_gpu
	engine.SetMapper(names.GonicMapper{})
	// go-sql-driver default loc is "UTC", namely the default timezone of returned time2.Time.
	// While xorm default timezone is "Local".
	// If they mismatch, xorm will overwrite the timezone by its own, hence messes up the original time2.
	// Fix by set them identical.
	if !strings.Contains(uri, "loc=") || strings.Contains(uri, "loc=UTC") {
		engine.SetTZDatabase(time.UTC)
		engine.SetTZLocation(time.UTC)
	} else if strings.Contains(uri, "loc=Local") {
		engine.SetTZDatabase(time.Local)
		engine.SetTZLocation(time.Local)
	} else {
		log.Warningf("Please make sure your 'loc' arg of sql driver won't cause any time2 parsing problem")
	}
	engine.SetLogger(NewHumanFriendlyLogger())
	engine.ShowSQL(showSQL)
	return engine
}

// InitMockAndDB initializes a mock db
func InitMockAndDB(showSQL bool) (*Database, sqlmock.Sqlmock, error) {
	mockDB, mock, err := sqlmock.New()
	if err != nil {
		return nil, nil, err
	}
	// Replace the underlying sql.DB with mock
	xormEngine, err := GetXormEngine(mockDB, showSQL)
	if err != nil {
		return nil, nil, err
	}
	db := &Database{Engine: xormEngine}
	return db, mock, nil
}

// GetXormEngine gets xorm engine
func GetXormEngine(db *sql.DB, showSQL bool) (*xorm.Engine, error) {
	xormEngine, err := xorm.NewEngine("mysql", "")
	if err != nil {
		return nil, err
	}
	// Replace the underlying sql.DB with mock
	xormEngine.DB().DB = db
	// Check availability
	err = xormEngine.Ping()
	if err != nil {
		return nil, err
	}
	xormEngine = postProcessEngine(xormEngine, showSQL)
	return xormEngine, nil
}

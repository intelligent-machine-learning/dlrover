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
	"fmt"
	"os"
	"strings"
	"xorm.io/xorm/log"
)

// MaxStringLength is the max length of string allowed to print to logger
const MaxStringLength = 128

// HumanFriendlyLogger struct
type HumanFriendlyLogger struct {
	log.ContextLogger
}

// NewHumanFriendlyLogger creates a new HumanFriendlyLogger struct
func NewHumanFriendlyLogger() *HumanFriendlyLogger {
	return &HumanFriendlyLogger{
		ContextLogger: log.NewLoggerAdapter(log.NewSimpleLogger(os.Stdout)),
	}
}

// AfterSQL overrides the default func to print human-friendly logs
func (l *HumanFriendlyLogger) AfterSQL(ctx log.LogContext) {
	var sessionPart string
	v := ctx.Ctx.Value(log.SessionIDKey)
	if key, ok := v.(string); ok {
		sessionPart = fmt.Sprintf(" [%s]", key)
	}

	// In string replacement
	sqlTemplate := strings.Replace(ctx.SQL, "?", "%v", -1)
	args := shortedStringInArgs(ctx.Args)
	sql := fmt.Sprintf(sqlTemplate, args...)

	if ctx.ExecuteTime > 0 {
		l.Infof("[SQL]%s %s - %v", sessionPart, sql, ctx.ExecuteTime)
	} else {
		l.Infof("[SQL]%s %s", sessionPart, sql)
	}
}

func shortedStringInArgs(args []interface{}) []interface{} {
	shortenArgs := make([]interface{}, 0, len(args))
	for _, arg := range args {
		if s, ok := arg.(string); ok {
			if len(s) > MaxStringLength {
				// If string is too long, it's meaningless to print it
				arg = "[...]"
			} else {
				// Make the printed sql copy, paste & run as is
				arg = fmt.Sprintf(`"%v"`, s)
			}
		}
		shortenArgs = append(shortenArgs, arg)
	}
	return shortenArgs
}

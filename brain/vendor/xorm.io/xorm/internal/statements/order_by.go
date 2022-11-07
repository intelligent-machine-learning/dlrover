// Copyright 2022 The Xorm Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package statements

import (
	"fmt"
	"strings"

	"xorm.io/builder"
)

func (statement *Statement) HasOrderBy() bool {
	return statement.orderStr != ""
}

// ResetOrderBy reset ordery conditions
func (statement *Statement) ResetOrderBy() {
	statement.orderStr = ""
	statement.orderArgs = nil
}

// WriteOrderBy write order by to writer
func (statement *Statement) WriteOrderBy(w builder.Writer) error {
	if len(statement.orderStr) > 0 {
		if _, err := fmt.Fprintf(w, " ORDER BY %s", statement.orderStr); err != nil {
			return err
		}
		w.Append(statement.orderArgs...)
	}
	return nil
}

// OrderBy generate "Order By order" statement
func (statement *Statement) OrderBy(order interface{}, args ...interface{}) *Statement {
	if len(statement.orderStr) > 0 {
		statement.orderStr += ", "
	}
	var rawOrder string
	switch t := order.(type) {
	case (*builder.Expression):
		rawOrder = t.Content()
		args = t.Args()
	case string:
		rawOrder = t
	default:
		statement.LastError = ErrUnSupportedSQLType
		return statement
	}
	statement.orderStr += statement.ReplaceQuote(rawOrder)
	if len(args) > 0 {
		statement.orderArgs = append(statement.orderArgs, args...)
	}
	return statement
}

// Desc generate `ORDER BY xx DESC`
func (statement *Statement) Desc(colNames ...string) *Statement {
	var buf strings.Builder
	if len(statement.orderStr) > 0 {
		fmt.Fprint(&buf, statement.orderStr, ", ")
	}
	for i, col := range colNames {
		if i > 0 {
			fmt.Fprint(&buf, ", ")
		}
		_ = statement.dialect.Quoter().QuoteTo(&buf, col)
		fmt.Fprint(&buf, " DESC")
	}
	statement.orderStr = buf.String()
	return statement
}

// Asc provide asc order by query condition, the input parameters are columns.
func (statement *Statement) Asc(colNames ...string) *Statement {
	var buf strings.Builder
	if len(statement.orderStr) > 0 {
		fmt.Fprint(&buf, statement.orderStr, ", ")
	}
	for i, col := range colNames {
		if i > 0 {
			fmt.Fprint(&buf, ", ")
		}
		_ = statement.dialect.Quoter().QuoteTo(&buf, col)
		fmt.Fprint(&buf, " ASC")
	}
	statement.orderStr = buf.String()
	return statement
}

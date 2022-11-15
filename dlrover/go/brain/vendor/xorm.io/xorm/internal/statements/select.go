// Copyright 2022 The Xorm Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package statements

import (
	"fmt"
	"strings"

	"xorm.io/xorm/schemas"
)

// Select replace select
func (statement *Statement) Select(str string) *Statement {
	statement.SelectStr = statement.ReplaceQuote(str)
	return statement
}

func col2NewCols(columns ...string) []string {
	newColumns := make([]string, 0, len(columns))
	for _, col := range columns {
		col = strings.Replace(col, "`", "", -1)
		col = strings.Replace(col, `"`, "", -1)
		ccols := strings.Split(col, ",")
		for _, c := range ccols {
			newColumns = append(newColumns, strings.TrimSpace(c))
		}
	}
	return newColumns
}

// Cols generate "col1, col2" statement
func (statement *Statement) Cols(columns ...string) *Statement {
	cols := col2NewCols(columns...)
	for _, nc := range cols {
		statement.ColumnMap.Add(nc)
	}
	return statement
}

// ColumnStr returns column string
func (statement *Statement) ColumnStr() string {
	return statement.dialect.Quoter().Join(statement.ColumnMap, ", ")
}

// AllCols update use only: update all columns
func (statement *Statement) AllCols() *Statement {
	statement.useAllCols = true
	return statement
}

// MustCols update use only: must update columns
func (statement *Statement) MustCols(columns ...string) *Statement {
	newColumns := col2NewCols(columns...)
	for _, nc := range newColumns {
		statement.MustColumnMap[strings.ToLower(nc)] = true
	}
	return statement
}

// UseBool indicates that use bool fields as update contents and query contiditions
func (statement *Statement) UseBool(columns ...string) *Statement {
	if len(columns) > 0 {
		statement.MustCols(columns...)
	} else {
		statement.allUseBool = true
	}
	return statement
}

// Omit do not use the columns
func (statement *Statement) Omit(columns ...string) {
	newColumns := col2NewCols(columns...)
	for _, nc := range newColumns {
		statement.OmitColumnMap = append(statement.OmitColumnMap, nc)
	}
}

func (statement *Statement) genColumnStr() string {
	if statement.RefTable == nil {
		return ""
	}

	var buf strings.Builder
	columns := statement.RefTable.Columns()

	for _, col := range columns {
		if statement.OmitColumnMap.Contain(col.Name) {
			continue
		}

		if len(statement.ColumnMap) > 0 && !statement.ColumnMap.Contain(col.Name) {
			continue
		}

		if col.MapType == schemas.ONLYTODB {
			continue
		}

		if buf.Len() != 0 {
			buf.WriteString(", ")
		}

		if statement.JoinStr != "" {
			if statement.TableAlias != "" {
				buf.WriteString(statement.TableAlias)
			} else {
				buf.WriteString(statement.TableName())
			}

			buf.WriteString(".")
		}

		statement.dialect.Quoter().QuoteTo(&buf, col.Name)
	}

	return buf.String()
}

func (statement *Statement) colName(col *schemas.Column, tableName string) string {
	if statement.needTableName() {
		nm := tableName
		if len(statement.TableAlias) > 0 {
			nm = statement.TableAlias
		}
		return fmt.Sprintf("%s.%s", statement.quote(nm), statement.quote(col.Name))
	}
	return statement.quote(col.Name)
}

// Distinct generates "DISTINCT col1, col2 " statement
func (statement *Statement) Distinct(columns ...string) *Statement {
	statement.IsDistinct = true
	statement.Cols(columns...)
	return statement
}

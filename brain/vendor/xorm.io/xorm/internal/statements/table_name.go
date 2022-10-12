// Copyright 2022 The Xorm Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package statements

import (
	"fmt"
	"strings"

	"xorm.io/builder"
	"xorm.io/xorm/schemas"
)

// TableName return current tableName
func (statement *Statement) TableName() string {
	if statement.AltTableName != "" {
		return statement.AltTableName
	}

	return statement.tableName
}

// Alias set the table alias
func (statement *Statement) Alias(alias string) *Statement {
	statement.TableAlias = alias
	return statement
}

func (statement *Statement) writeAlias(w builder.Writer) error {
	if statement.TableAlias != "" {
		if statement.dialect.URI().DBType == schemas.ORACLE {
			if _, err := fmt.Fprint(w, " ", statement.quote(statement.TableAlias)); err != nil {
				return err
			}
		} else {
			if _, err := fmt.Fprint(w, " AS ", statement.quote(statement.TableAlias)); err != nil {
				return err
			}
		}
	}
	return nil
}

func (statement *Statement) writeTableName(w builder.Writer) error {
	if statement.dialect.URI().DBType == schemas.MSSQL && strings.Contains(statement.TableName(), "..") {
		if _, err := fmt.Fprint(w, statement.TableName()); err != nil {
			return err
		}
	} else {
		if _, err := fmt.Fprint(w, statement.quote(statement.TableName())); err != nil {
			return err
		}
	}
	return nil
}

// Copyright 2022 The Xorm Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package statements

import (
	"fmt"
	"strings"

	"xorm.io/builder"
	"xorm.io/xorm/dialects"
	"xorm.io/xorm/internal/utils"
	"xorm.io/xorm/schemas"
)

// Join The joinOP should be one of INNER, LEFT OUTER, CROSS etc - this will be prepended to JOIN
func (statement *Statement) Join(joinOP string, tablename interface{}, condition string, args ...interface{}) *Statement {
	var buf strings.Builder
	if len(statement.JoinStr) > 0 {
		fmt.Fprintf(&buf, "%v %v JOIN ", statement.JoinStr, joinOP)
	} else {
		fmt.Fprintf(&buf, "%v JOIN ", joinOP)
	}

	switch tp := tablename.(type) {
	case builder.Builder:
		subSQL, subQueryArgs, err := tp.ToSQL()
		if err != nil {
			statement.LastError = err
			return statement
		}

		fields := strings.Split(tp.TableName(), ".")
		aliasName := statement.dialect.Quoter().Trim(fields[len(fields)-1])
		aliasName = schemas.CommonQuoter.Trim(aliasName)

		fmt.Fprintf(&buf, "(%s) %s ON %v", statement.ReplaceQuote(subSQL), statement.quote(aliasName), statement.ReplaceQuote(condition))
		statement.joinArgs = append(statement.joinArgs, subQueryArgs...)
	case *builder.Builder:
		subSQL, subQueryArgs, err := tp.ToSQL()
		if err != nil {
			statement.LastError = err
			return statement
		}

		fields := strings.Split(tp.TableName(), ".")
		aliasName := statement.dialect.Quoter().Trim(fields[len(fields)-1])
		aliasName = schemas.CommonQuoter.Trim(aliasName)

		fmt.Fprintf(&buf, "(%s) %s ON %v", statement.ReplaceQuote(subSQL), statement.quote(aliasName), statement.ReplaceQuote(condition))
		statement.joinArgs = append(statement.joinArgs, subQueryArgs...)
	default:
		tbName := dialects.FullTableName(statement.dialect, statement.tagParser.GetTableMapper(), tablename, true)
		if !utils.IsSubQuery(tbName) {
			var buf strings.Builder
			_ = statement.dialect.Quoter().QuoteTo(&buf, tbName)
			tbName = buf.String()
		} else {
			tbName = statement.ReplaceQuote(tbName)
		}
		fmt.Fprintf(&buf, "%s ON %v", tbName, statement.ReplaceQuote(condition))
	}

	statement.JoinStr = buf.String()
	statement.joinArgs = append(statement.joinArgs, args...)
	return statement
}

func (statement *Statement) writeJoin(w builder.Writer) error {
	if statement.JoinStr != "" {
		if _, err := fmt.Fprint(w, " ", statement.JoinStr); err != nil {
			return err
		}
		w.Append(statement.joinArgs...)
	}
	return nil
}

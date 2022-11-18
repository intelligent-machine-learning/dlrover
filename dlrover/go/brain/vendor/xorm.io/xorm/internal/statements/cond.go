// Copyright 2022 The Xorm Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package statements

import (
	"xorm.io/builder"
	"xorm.io/xorm/schemas"
)

type QuoteReplacer struct {
	*builder.BytesWriter
	quoter schemas.Quoter
}

func (q *QuoteReplacer) Write(p []byte) (n int, err error) {
	c := q.quoter.Replace(string(p))
	return q.BytesWriter.Builder.WriteString(c)
}

func (statement *Statement) QuoteReplacer(w *builder.BytesWriter) *QuoteReplacer {
	return &QuoteReplacer{
		BytesWriter: w,
		quoter:      statement.dialect.Quoter(),
	}
}

// Where add Where statement
func (statement *Statement) Where(query interface{}, args ...interface{}) *Statement {
	return statement.And(query, args...)
}

// And add Where & and statement
func (statement *Statement) And(query interface{}, args ...interface{}) *Statement {
	switch qr := query.(type) {
	case string:
		cond := builder.Expr(qr, args...)
		statement.cond = statement.cond.And(cond)
	case map[string]interface{}:
		cond := make(builder.Eq)
		for k, v := range qr {
			cond[statement.quote(k)] = v
		}
		statement.cond = statement.cond.And(cond)
	case builder.Cond:
		statement.cond = statement.cond.And(qr)
		for _, v := range args {
			if vv, ok := v.(builder.Cond); ok {
				statement.cond = statement.cond.And(vv)
			}
		}
	default:
		statement.LastError = ErrConditionType
	}

	return statement
}

// Or add Where & Or statement
func (statement *Statement) Or(query interface{}, args ...interface{}) *Statement {
	switch qr := query.(type) {
	case string:
		cond := builder.Expr(qr, args...)
		statement.cond = statement.cond.Or(cond)
	case map[string]interface{}:
		cond := make(builder.Eq)
		for k, v := range qr {
			cond[statement.quote(k)] = v
		}
		statement.cond = statement.cond.Or(cond)
	case builder.Cond:
		statement.cond = statement.cond.Or(qr)
		for _, v := range args {
			if vv, ok := v.(builder.Cond); ok {
				statement.cond = statement.cond.Or(vv)
			}
		}
	default:
		statement.LastError = ErrConditionType
	}
	return statement
}

// In generate "Where column IN (?) " statement
func (statement *Statement) In(column string, args ...interface{}) *Statement {
	in := builder.In(statement.quote(column), args...)
	statement.cond = statement.cond.And(in)
	return statement
}

// NotIn generate "Where column NOT IN (?) " statement
func (statement *Statement) NotIn(column string, args ...interface{}) *Statement {
	notIn := builder.NotIn(statement.quote(column), args...)
	statement.cond = statement.cond.And(notIn)
	return statement
}

// SetNoAutoCondition if you do not want convert bean's field as query condition, then use this function
func (statement *Statement) SetNoAutoCondition(no ...bool) *Statement {
	statement.NoAutoCondition = true
	if len(no) > 0 {
		statement.NoAutoCondition = no[0]
	}
	return statement
}

// Conds returns condtions
func (statement *Statement) Conds() builder.Cond {
	return statement.cond
}

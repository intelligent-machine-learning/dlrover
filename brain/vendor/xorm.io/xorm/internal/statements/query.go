// Copyright 2019 The Xorm Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package statements

import (
	"errors"
	"fmt"
	"reflect"
	"strings"

	"xorm.io/builder"
	"xorm.io/xorm/internal/utils"
	"xorm.io/xorm/schemas"
)

// GenQuerySQL generate query SQL
func (statement *Statement) GenQuerySQL(sqlOrArgs ...interface{}) (string, []interface{}, error) {
	if len(sqlOrArgs) > 0 {
		return statement.ConvertSQLOrArgs(sqlOrArgs...)
	}

	if statement.RawSQL != "" {
		return statement.GenRawSQL(), statement.RawParams, nil
	}

	if len(statement.TableName()) <= 0 {
		return "", nil, ErrTableNotFound
	}

	columnStr := statement.ColumnStr()
	if len(statement.SelectStr) > 0 {
		columnStr = statement.SelectStr
	} else {
		if statement.JoinStr == "" {
			if columnStr == "" {
				if statement.GroupByStr != "" {
					columnStr = statement.quoteColumnStr(statement.GroupByStr)
				} else {
					columnStr = statement.genColumnStr()
				}
			}
		} else {
			if columnStr == "" {
				if statement.GroupByStr != "" {
					columnStr = statement.quoteColumnStr(statement.GroupByStr)
				} else {
					columnStr = "*"
				}
			}
		}
		if columnStr == "" {
			columnStr = "*"
		}
	}

	if err := statement.ProcessIDParam(); err != nil {
		return "", nil, err
	}

	return statement.genSelectSQL(columnStr, true, true)
}

// GenSumSQL generates sum SQL
func (statement *Statement) GenSumSQL(bean interface{}, columns ...string) (string, []interface{}, error) {
	if statement.RawSQL != "" {
		return statement.GenRawSQL(), statement.RawParams, nil
	}

	if err := statement.SetRefBean(bean); err != nil {
		return "", nil, err
	}

	sumStrs := make([]string, 0, len(columns))
	for _, colName := range columns {
		if !strings.Contains(colName, " ") && !strings.Contains(colName, "(") {
			colName = statement.quote(colName)
		} else {
			colName = statement.ReplaceQuote(colName)
		}
		sumStrs = append(sumStrs, fmt.Sprintf("COALESCE(sum(%s),0)", colName))
	}
	sumSelect := strings.Join(sumStrs, ", ")

	if err := statement.MergeConds(bean); err != nil {
		return "", nil, err
	}

	return statement.genSelectSQL(sumSelect, true, true)
}

// GenGetSQL generates Get SQL
func (statement *Statement) GenGetSQL(bean interface{}) (string, []interface{}, error) {
	var isStruct bool
	if bean != nil {
		v := rValue(bean)
		isStruct = v.Kind() == reflect.Struct
		if isStruct {
			if err := statement.SetRefBean(bean); err != nil {
				return "", nil, err
			}
		}
	}

	columnStr := statement.ColumnStr()
	if len(statement.SelectStr) > 0 {
		columnStr = statement.SelectStr
	} else {
		// TODO: always generate column names, not use * even if join
		if len(statement.JoinStr) == 0 {
			if len(columnStr) == 0 {
				if len(statement.GroupByStr) > 0 {
					columnStr = statement.quoteColumnStr(statement.GroupByStr)
				} else {
					columnStr = statement.genColumnStr()
				}
			}
		} else {
			if len(columnStr) == 0 {
				if len(statement.GroupByStr) > 0 {
					columnStr = statement.quoteColumnStr(statement.GroupByStr)
				}
			}
		}
	}

	if len(columnStr) == 0 {
		columnStr = "*"
	}

	if isStruct {
		if err := statement.MergeConds(bean); err != nil {
			return "", nil, err
		}
	} else {
		if err := statement.ProcessIDParam(); err != nil {
			return "", nil, err
		}
	}

	return statement.genSelectSQL(columnStr, true, true)
}

// GenCountSQL generates the SQL for counting
func (statement *Statement) GenCountSQL(beans ...interface{}) (string, []interface{}, error) {
	if statement.RawSQL != "" {
		return statement.GenRawSQL(), statement.RawParams, nil
	}

	var condArgs []interface{}
	var err error
	if len(beans) > 0 {
		if err := statement.SetRefBean(beans[0]); err != nil {
			return "", nil, err
		}
		if err := statement.MergeConds(beans[0]); err != nil {
			return "", nil, err
		}
	}

	selectSQL := statement.SelectStr
	if len(selectSQL) <= 0 {
		if statement.IsDistinct {
			selectSQL = fmt.Sprintf("count(DISTINCT %s)", statement.ColumnStr())
		} else if statement.ColumnStr() != "" {
			selectSQL = fmt.Sprintf("count(%s)", statement.ColumnStr())
		} else {
			selectSQL = "count(*)"
		}
	}
	var subQuerySelect string
	if statement.GroupByStr != "" {
		subQuerySelect = statement.GroupByStr
	} else {
		subQuerySelect = selectSQL
	}

	sqlStr, condArgs, err := statement.genSelectSQL(subQuerySelect, false, false)
	if err != nil {
		return "", nil, err
	}

	if statement.GroupByStr != "" {
		sqlStr = fmt.Sprintf("SELECT %s FROM (%s) sub", selectSQL, sqlStr)
	}

	return sqlStr, condArgs, nil
}

func (statement *Statement) writeFrom(w builder.Writer) error {
	if _, err := fmt.Fprint(w, " FROM "); err != nil {
		return err
	}
	if err := statement.writeTableName(w); err != nil {
		return err
	}
	if err := statement.writeAlias(w); err != nil {
		return err
	}
	return statement.writeJoin(w)
}

func (statement *Statement) writeLimitOffset(w builder.Writer) error {
	if statement.Start > 0 {
		if statement.LimitN != nil {
			_, err := fmt.Fprintf(w, " LIMIT %v OFFSET %v", *statement.LimitN, statement.Start)
			return err
		}
		_, err := fmt.Fprintf(w, " LIMIT 0 OFFSET %v", statement.Start)
		return err
	}
	if statement.LimitN != nil {
		_, err := fmt.Fprint(w, " LIMIT ", *statement.LimitN)
		return err
	}
	// no limit statement
	return nil
}

func (statement *Statement) genSelectSQL(columnStr string, needLimit, needOrderBy bool) (string, []interface{}, error) {
	var (
		distinct      string
		dialect       = statement.dialect
		top, whereStr string
		mssqlCondi    = builder.NewWriter()
	)

	if statement.IsDistinct && !strings.HasPrefix(columnStr, "count") {
		distinct = "DISTINCT "
	}

	condWriter := builder.NewWriter()
	if err := statement.cond.WriteTo(statement.QuoteReplacer(condWriter)); err != nil {
		return "", nil, err
	}

	if condWriter.Len() > 0 {
		whereStr = " WHERE "
	}

	pLimitN := statement.LimitN
	if dialect.URI().DBType == schemas.MSSQL {
		if pLimitN != nil {
			LimitNValue := *pLimitN
			top = fmt.Sprintf("TOP %d ", LimitNValue)
		}
		if statement.Start > 0 {
			if statement.RefTable == nil {
				return "", nil, errors.New("Unsupported query limit without reference table")
			}
			var column string
			if len(statement.RefTable.PKColumns()) == 0 {
				for _, index := range statement.RefTable.Indexes {
					if len(index.Cols) == 1 {
						column = index.Cols[0]
						break
					}
				}
				if len(column) == 0 {
					column = statement.RefTable.ColumnsSeq()[0]
				}
			} else {
				column = statement.RefTable.PKColumns()[0].Name
			}
			if statement.needTableName() {
				if len(statement.TableAlias) > 0 {
					column = fmt.Sprintf("%s.%s", statement.TableAlias, column)
				} else {
					column = fmt.Sprintf("%s.%s", statement.TableName(), column)
				}
			}

			if _, err := fmt.Fprintf(mssqlCondi, "(%s NOT IN (SELECT TOP %d %s",
				column, statement.Start, column); err != nil {
				return "", nil, err
			}
			if err := statement.writeFrom(mssqlCondi); err != nil {
				return "", nil, err
			}
			if whereStr != "" {
				if _, err := fmt.Fprint(mssqlCondi, whereStr); err != nil {
					return "", nil, err
				}
				if err := utils.WriteBuilder(mssqlCondi, statement.QuoteReplacer(condWriter)); err != nil {
					return "", nil, err
				}
			}
			if needOrderBy {
				if err := statement.WriteOrderBy(mssqlCondi); err != nil {
					return "", nil, err
				}
			}
			if err := statement.WriteGroupBy(mssqlCondi); err != nil {
				return "", nil, err
			}
			if _, err := fmt.Fprint(mssqlCondi, "))"); err != nil {
				return "", nil, err
			}
		}
	}

	buf := builder.NewWriter()
	if _, err := fmt.Fprintf(buf, "SELECT %v%v%v", distinct, top, columnStr); err != nil {
		return "", nil, err
	}
	if err := statement.writeFrom(buf); err != nil {
		return "", nil, err
	}
	if whereStr != "" {
		if _, err := fmt.Fprint(buf, whereStr); err != nil {
			return "", nil, err
		}
		if err := utils.WriteBuilder(buf, statement.QuoteReplacer(condWriter)); err != nil {
			return "", nil, err
		}
	}
	if mssqlCondi.Len() > 0 {
		if len(whereStr) > 0 {
			if _, err := fmt.Fprint(buf, " AND "); err != nil {
				return "", nil, err
			}
		} else {
			if _, err := fmt.Fprint(buf, " WHERE "); err != nil {
				return "", nil, err
			}
		}

		if err := utils.WriteBuilder(buf, mssqlCondi); err != nil {
			return "", nil, err
		}
	}

	if err := statement.WriteGroupBy(buf); err != nil {
		return "", nil, err
	}
	if err := statement.writeHaving(buf); err != nil {
		return "", nil, err
	}
	if needOrderBy {
		if err := statement.WriteOrderBy(buf); err != nil {
			return "", nil, err
		}
	}
	if needLimit {
		if dialect.URI().DBType != schemas.MSSQL && dialect.URI().DBType != schemas.ORACLE {
			if err := statement.writeLimitOffset(buf); err != nil {
				return "", nil, err
			}
		} else if dialect.URI().DBType == schemas.ORACLE {
			if pLimitN != nil {
				oldString := buf.String()
				buf.Reset()
				rawColStr := columnStr
				if rawColStr == "*" {
					rawColStr = "at.*"
				}
				fmt.Fprintf(buf, "SELECT %v FROM (SELECT %v,ROWNUM RN FROM (%v) at WHERE ROWNUM <= %d) aat WHERE RN > %d",
					columnStr, rawColStr, oldString, statement.Start+*pLimitN, statement.Start)
			}
		}
	}
	if statement.IsForUpdate {
		return dialect.ForUpdateSQL(buf.String()), buf.Args(), nil
	}

	return buf.String(), buf.Args(), nil
}

// GenExistSQL generates Exist SQL
func (statement *Statement) GenExistSQL(bean ...interface{}) (string, []interface{}, error) {
	if statement.RawSQL != "" {
		return statement.GenRawSQL(), statement.RawParams, nil
	}

	var b interface{}
	if len(bean) > 0 {
		b = bean[0]
		beanValue := reflect.ValueOf(bean[0])
		if beanValue.Kind() != reflect.Ptr {
			return "", nil, errors.New("needs a pointer")
		}

		if beanValue.Elem().Kind() == reflect.Struct {
			if err := statement.SetRefBean(bean[0]); err != nil {
				return "", nil, err
			}
		}
	}
	tableName := statement.TableName()
	if len(tableName) <= 0 {
		return "", nil, ErrTableNotFound
	}
	if statement.RefTable != nil {
		return statement.Limit(1).GenGetSQL(b)
	}

	tableName = statement.quote(tableName)

	buf := builder.NewWriter()
	if statement.dialect.URI().DBType == schemas.MSSQL {
		if _, err := fmt.Fprintf(buf, "SELECT TOP 1 * FROM %s", tableName); err != nil {
			return "", nil, err
		}
		if err := statement.writeJoin(buf); err != nil {
			return "", nil, err
		}
		if statement.Conds().IsValid() {
			if _, err := fmt.Fprintf(buf, " WHERE "); err != nil {
				return "", nil, err
			}
			if err := statement.Conds().WriteTo(statement.QuoteReplacer(buf)); err != nil {
				return "", nil, err
			}
		}
	} else if statement.dialect.URI().DBType == schemas.ORACLE {
		if _, err := fmt.Fprintf(buf, "SELECT * FROM %s", tableName); err != nil {
			return "", nil, err
		}
		if err := statement.writeJoin(buf); err != nil {
			return "", nil, err
		}
		if _, err := fmt.Fprintf(buf, " WHERE "); err != nil {
			return "", nil, err
		}
		if statement.Conds().IsValid() {
			if err := statement.Conds().WriteTo(statement.QuoteReplacer(buf)); err != nil {
				return "", nil, err
			}
			if _, err := fmt.Fprintf(buf, " AND "); err != nil {
				return "", nil, err
			}
		}
		if _, err := fmt.Fprintf(buf, "ROWNUM=1"); err != nil {
			return "", nil, err
		}
	} else {
		if _, err := fmt.Fprintf(buf, "SELECT 1 FROM %s", tableName); err != nil {
			return "", nil, err
		}
		if err := statement.writeJoin(buf); err != nil {
			return "", nil, err
		}
		if statement.Conds().IsValid() {
			if _, err := fmt.Fprintf(buf, " WHERE "); err != nil {
				return "", nil, err
			}
			if err := statement.Conds().WriteTo(statement.QuoteReplacer(buf)); err != nil {
				return "", nil, err
			}
		}
		if _, err := fmt.Fprintf(buf, " LIMIT 1"); err != nil {
			return "", nil, err
		}
	}

	return buf.String(), buf.Args(), nil
}

// GenFindSQL generates Find SQL
func (statement *Statement) GenFindSQL(autoCond builder.Cond) (string, []interface{}, error) {
	if statement.RawSQL != "" {
		return statement.GenRawSQL(), statement.RawParams, nil
	}

	if len(statement.TableName()) <= 0 {
		return "", nil, ErrTableNotFound
	}

	columnStr := statement.ColumnStr()
	if len(statement.SelectStr) > 0 {
		columnStr = statement.SelectStr
	} else {
		if statement.JoinStr == "" {
			if columnStr == "" {
				if statement.GroupByStr != "" {
					columnStr = statement.quoteColumnStr(statement.GroupByStr)
				} else {
					columnStr = statement.genColumnStr()
				}
			}
		} else {
			if columnStr == "" {
				if statement.GroupByStr != "" {
					columnStr = statement.quoteColumnStr(statement.GroupByStr)
				} else {
					columnStr = "*"
				}
			}
		}
		if columnStr == "" {
			columnStr = "*"
		}
	}

	statement.cond = statement.cond.And(autoCond)

	return statement.genSelectSQL(columnStr, true, true)
}

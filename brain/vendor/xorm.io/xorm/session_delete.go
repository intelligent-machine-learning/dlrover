// Copyright 2016 The Xorm Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package xorm

import (
	"errors"
	"fmt"
	"strconv"

	"xorm.io/builder"
	"xorm.io/xorm/caches"
	"xorm.io/xorm/internal/utils"
	"xorm.io/xorm/schemas"
)

var (
	// ErrNeedDeletedCond delete needs less one condition error
	ErrNeedDeletedCond = errors.New("Delete action needs at least one condition")

	// ErrNotImplemented not implemented
	ErrNotImplemented = errors.New("Not implemented")
)

func (session *Session) cacheDelete(table *schemas.Table, tableName, sqlStr string, args ...interface{}) error {
	if table == nil ||
		session.tx != nil {
		return ErrCacheFailed
	}

	for _, filter := range session.engine.dialect.Filters() {
		sqlStr = filter.Do(sqlStr)
	}

	newsql := session.statement.ConvertIDSQL(sqlStr)
	if newsql == "" {
		return ErrCacheFailed
	}

	cacher := session.engine.cacherMgr.GetCacher(tableName)
	pkColumns := table.PKColumns()
	ids, err := caches.GetCacheSql(cacher, tableName, newsql, args)
	if err != nil {
		rows, err := session.queryRows(newsql, args...)
		if err != nil {
			return err
		}
		defer rows.Close()

		resultsSlice, err := session.engine.ScanStringMaps(rows)
		if err != nil {
			return err
		}
		ids = make([]schemas.PK, 0)
		if len(resultsSlice) > 0 {
			for _, data := range resultsSlice {
				var id int64
				var pk schemas.PK = make([]interface{}, 0)
				for _, col := range pkColumns {
					if v, ok := data[col.Name]; !ok {
						return errors.New("no id")
					} else if col.SQLType.IsText() {
						pk = append(pk, v)
					} else if col.SQLType.IsNumeric() {
						id, err = strconv.ParseInt(v, 10, 64)
						if err != nil {
							return err
						}
						pk = append(pk, id)
					} else {
						return errors.New("not supported primary key type")
					}
				}
				ids = append(ids, pk)
			}
		}
	}

	for _, id := range ids {
		session.engine.logger.Debugf("[cache] delete cache obj: %v, %v", tableName, id)
		sid, err := id.ToString()
		if err != nil {
			return err
		}
		cacher.DelBean(tableName, sid)
	}
	session.engine.logger.Debugf("[cache] clear cache table: %v", tableName)
	cacher.ClearIds(tableName)
	return nil
}

// Delete records, bean's non-empty fields are conditions
func (session *Session) Delete(beans ...interface{}) (int64, error) {
	if session.isAutoClose {
		defer session.Close()
	}

	if session.statement.LastError != nil {
		return 0, session.statement.LastError
	}

	var (
		condWriter = builder.NewWriter()
		err        error
		bean       interface{}
	)
	if len(beans) > 0 {
		bean = beans[0]
		if err = session.statement.SetRefBean(bean); err != nil {
			return 0, err
		}

		executeBeforeClosures(session, bean)

		if processor, ok := interface{}(bean).(BeforeDeleteProcessor); ok {
			processor.BeforeDelete()
		}

		if err = session.statement.MergeConds(bean); err != nil {
			return 0, err
		}
	}

	if err = session.statement.Conds().WriteTo(session.statement.QuoteReplacer(condWriter)); err != nil {
		return 0, err
	}

	pLimitN := session.statement.LimitN
	if condWriter.Len() == 0 && (pLimitN == nil || *pLimitN == 0) {
		return 0, ErrNeedDeletedCond
	}

	tableNameNoQuote := session.statement.TableName()
	tableName := session.engine.Quote(tableNameNoQuote)
	table := session.statement.RefTable
	deleteSQLWriter := builder.NewWriter()
	fmt.Fprintf(deleteSQLWriter, "DELETE FROM %v", tableName)
	if condWriter.Len() > 0 {
		fmt.Fprintf(deleteSQLWriter, " WHERE %v", condWriter.String())
		deleteSQLWriter.Append(condWriter.Args()...)
	}

	orderSQLWriter := builder.NewWriter()
	if err := session.statement.WriteOrderBy(orderSQLWriter); err != nil {
		return 0, err
	}

	if pLimitN != nil && *pLimitN > 0 {
		limitNValue := *pLimitN
		if _, err := fmt.Fprintf(orderSQLWriter, " LIMIT %d", limitNValue); err != nil {
			return 0, err
		}
	}

	orderCondWriter := builder.NewWriter()
	if orderSQLWriter.Len() > 0 {
		switch session.engine.dialect.URI().DBType {
		case schemas.POSTGRES:
			if condWriter.Len() > 0 {
				fmt.Fprintf(orderCondWriter, " AND ")
			} else {
				fmt.Fprintf(orderCondWriter, " WHERE ")
			}
			fmt.Fprintf(orderCondWriter, "ctid IN (SELECT ctid FROM %s%s)", tableName, orderSQLWriter.String())
			orderCondWriter.Append(orderSQLWriter.Args()...)
		case schemas.SQLITE:
			if condWriter.Len() > 0 {
				fmt.Fprintf(orderCondWriter, " AND ")
			} else {
				fmt.Fprintf(orderCondWriter, " WHERE ")
			}
			fmt.Fprintf(orderCondWriter, "rowid IN (SELECT rowid FROM %s%s)", tableName, orderSQLWriter.String())
			// TODO: how to handle delete limit on mssql?
		case schemas.MSSQL:
			return 0, ErrNotImplemented
		default:
			fmt.Fprint(orderCondWriter, orderSQLWriter.String())
			orderCondWriter.Append(orderSQLWriter.Args()...)
		}
	}

	realSQLWriter := builder.NewWriter()
	argsForCache := make([]interface{}, 0, len(deleteSQLWriter.Args())*2)
	copy(argsForCache, deleteSQLWriter.Args())
	argsForCache = append(deleteSQLWriter.Args(), argsForCache...)
	if session.statement.GetUnscoped() || table == nil || table.DeletedColumn() == nil { // tag "deleted" is disabled
		if err := utils.WriteBuilder(realSQLWriter, deleteSQLWriter, orderCondWriter); err != nil {
			return 0, err
		}
	} else {
		deletedColumn := table.DeletedColumn()
		if _, err := fmt.Fprintf(realSQLWriter, "UPDATE %v SET %v = ? WHERE %v",
			session.engine.Quote(session.statement.TableName()),
			session.engine.Quote(deletedColumn.Name),
			condWriter.String()); err != nil {
			return 0, err
		}
		val, t, err := session.engine.nowTime(deletedColumn)
		if err != nil {
			return 0, err
		}
		realSQLWriter.Append(val)
		realSQLWriter.Append(condWriter.Args()...)

		if err := utils.WriteBuilder(realSQLWriter, orderCondWriter); err != nil {
			return 0, err
		}

		colName := deletedColumn.Name
		session.afterClosures = append(session.afterClosures, func(bean interface{}) {
			col := table.GetColumn(colName)
			setColumnTime(bean, col, t)
		})
	}

	if cacher := session.engine.GetCacher(tableNameNoQuote); cacher != nil && session.statement.UseCache {
		_ = session.cacheDelete(table, tableNameNoQuote, deleteSQLWriter.String(), argsForCache...)
	}

	session.statement.RefTable = table
	res, err := session.exec(realSQLWriter.String(), realSQLWriter.Args()...)
	if err != nil {
		return 0, err
	}

	if bean != nil {
		// handle after delete processors
		if session.isAutoCommit {
			for _, closure := range session.afterClosures {
				closure(bean)
			}
			if processor, ok := interface{}(bean).(AfterDeleteProcessor); ok {
				processor.AfterDelete()
			}
		} else {
			lenAfterClosures := len(session.afterClosures)
			if lenAfterClosures > 0 && len(beans) > 0 {
				if value, has := session.afterDeleteBeans[beans[0]]; has && value != nil {
					*value = append(*value, session.afterClosures...)
				} else {
					afterClosures := make([]func(interface{}), lenAfterClosures)
					copy(afterClosures, session.afterClosures)
					session.afterDeleteBeans[bean] = &afterClosures
				}
			} else {
				if _, ok := interface{}(bean).(AfterDeleteProcessor); ok {
					session.afterDeleteBeans[bean] = nil
				}
			}
		}
	}
	cleanupProcessorsClosures(&session.afterClosures)
	// --

	return res.RowsAffected()
}

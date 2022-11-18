// Copyright 2015 The Xorm Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package statements

import (
	"database/sql/driver"
	"errors"
	"fmt"
	"math/big"
	"reflect"
	"strings"
	"time"

	"xorm.io/builder"
	"xorm.io/xorm/contexts"
	"xorm.io/xorm/convert"
	"xorm.io/xorm/dialects"
	"xorm.io/xorm/internal/json"
	"xorm.io/xorm/internal/utils"
	"xorm.io/xorm/schemas"
	"xorm.io/xorm/tags"
)

var (
	// ErrConditionType condition type unsupported
	ErrConditionType = errors.New("Unsupported condition type")
	// ErrUnSupportedSQLType parameter of SQL is not supported
	ErrUnSupportedSQLType = errors.New("Unsupported sql type")
	// ErrUnSupportedType unsupported error
	ErrUnSupportedType = errors.New("Unsupported type error")
	// ErrTableNotFound table not found error
	ErrTableNotFound = errors.New("Table not found")
)

// Statement save all the sql info for executing SQL
type Statement struct {
	RefTable        *schemas.Table
	dialect         dialects.Dialect
	defaultTimeZone *time.Location
	tagParser       *tags.Parser
	Start           int
	LimitN          *int
	idParam         schemas.PK
	orderStr        string
	orderArgs       []interface{}
	JoinStr         string
	joinArgs        []interface{}
	GroupByStr      string
	HavingStr       string
	SelectStr       string
	useAllCols      bool
	AltTableName    string
	tableName       string
	RawSQL          string
	RawParams       []interface{}
	UseCascade      bool
	UseAutoJoin     bool
	StoreEngine     string
	Charset         string
	UseCache        bool
	UseAutoTime     bool
	NoAutoCondition bool
	IsDistinct      bool
	IsForUpdate     bool
	TableAlias      string
	allUseBool      bool
	CheckVersion    bool
	unscoped        bool
	ColumnMap       columnMap
	OmitColumnMap   columnMap
	MustColumnMap   map[string]bool
	NullableMap     map[string]bool
	IncrColumns     exprParams
	DecrColumns     exprParams
	ExprColumns     exprParams
	cond            builder.Cond
	BufferSize      int
	Context         contexts.ContextCache
	LastError       error
}

// NewStatement creates a new statement
func NewStatement(dialect dialects.Dialect, tagParser *tags.Parser, defaultTimeZone *time.Location) *Statement {
	statement := &Statement{
		dialect:         dialect,
		tagParser:       tagParser,
		defaultTimeZone: defaultTimeZone,
	}
	statement.Reset()
	return statement
}

// SetTableName set table name
func (statement *Statement) SetTableName(tableName string) {
	statement.tableName = tableName
}

// GenRawSQL generates correct raw sql
func (statement *Statement) GenRawSQL() string {
	return statement.ReplaceQuote(statement.RawSQL)
}

// ReplaceQuote replace sql key words with quote
func (statement *Statement) ReplaceQuote(sql string) string {
	if sql == "" || statement.dialect.URI().DBType == schemas.MYSQL ||
		statement.dialect.URI().DBType == schemas.SQLITE {
		return sql
	}
	return statement.dialect.Quoter().Replace(sql)
}

// SetContextCache sets context cache
func (statement *Statement) SetContextCache(ctxCache contexts.ContextCache) {
	statement.Context = ctxCache
}

// Reset reset all the statement's fields
func (statement *Statement) Reset() {
	statement.RefTable = nil
	statement.Start = 0
	statement.LimitN = nil
	statement.ResetOrderBy()
	statement.UseCascade = true
	statement.JoinStr = ""
	statement.joinArgs = make([]interface{}, 0)
	statement.GroupByStr = ""
	statement.HavingStr = ""
	statement.ColumnMap = columnMap{}
	statement.OmitColumnMap = columnMap{}
	statement.AltTableName = ""
	statement.tableName = ""
	statement.idParam = nil
	statement.RawSQL = ""
	statement.RawParams = make([]interface{}, 0)
	statement.UseCache = true
	statement.UseAutoTime = true
	statement.NoAutoCondition = false
	statement.IsDistinct = false
	statement.IsForUpdate = false
	statement.TableAlias = ""
	statement.SelectStr = ""
	statement.allUseBool = false
	statement.useAllCols = false
	statement.MustColumnMap = make(map[string]bool)
	statement.NullableMap = make(map[string]bool)
	statement.CheckVersion = true
	statement.unscoped = false
	statement.IncrColumns = exprParams{}
	statement.DecrColumns = exprParams{}
	statement.ExprColumns = exprParams{}
	statement.cond = builder.NewCond()
	statement.BufferSize = 0
	statement.Context = nil
	statement.LastError = nil
}

// SQL adds raw sql statement
func (statement *Statement) SQL(query interface{}, args ...interface{}) *Statement {
	switch query.(type) {
	case (*builder.Builder):
		var err error
		statement.RawSQL, statement.RawParams, err = query.(*builder.Builder).ToSQL()
		if err != nil {
			statement.LastError = err
		}
	case string:
		statement.RawSQL = query.(string)
		statement.RawParams = args
	default:
		statement.LastError = ErrUnSupportedSQLType
	}

	return statement
}

func (statement *Statement) quote(s string) string {
	return statement.dialect.Quoter().Quote(s)
}

// SetRefValue set ref value
func (statement *Statement) SetRefValue(v reflect.Value) error {
	var err error
	statement.RefTable, err = statement.tagParser.ParseWithCache(reflect.Indirect(v))
	if err != nil {
		return err
	}
	statement.tableName = dialects.FullTableName(statement.dialect, statement.tagParser.GetTableMapper(), v, true)
	return nil
}

func rValue(bean interface{}) reflect.Value {
	return reflect.Indirect(reflect.ValueOf(bean))
}

// SetRefBean set ref bean
func (statement *Statement) SetRefBean(bean interface{}) error {
	var err error
	statement.RefTable, err = statement.tagParser.ParseWithCache(rValue(bean))
	if err != nil {
		return err
	}
	statement.tableName = dialects.FullTableName(statement.dialect, statement.tagParser.GetTableMapper(), bean, true)
	return nil
}

func (statement *Statement) needTableName() bool {
	return len(statement.JoinStr) > 0
}

// Incr Generate  "Update ... Set column = column + arg" statement
func (statement *Statement) Incr(column string, arg ...interface{}) *Statement {
	if len(arg) > 0 {
		statement.IncrColumns.Add(column, arg[0])
	} else {
		statement.IncrColumns.Add(column, 1)
	}
	return statement
}

// Decr Generate  "Update ... Set column = column - arg" statement
func (statement *Statement) Decr(column string, arg ...interface{}) *Statement {
	if len(arg) > 0 {
		statement.DecrColumns.Add(column, arg[0])
	} else {
		statement.DecrColumns.Add(column, 1)
	}
	return statement
}

// SetExpr Generate  "Update ... Set column = {expression}" statement
func (statement *Statement) SetExpr(column string, expression interface{}) *Statement {
	if e, ok := expression.(string); ok {
		statement.ExprColumns.Add(column, statement.dialect.Quoter().Replace(e))
	} else {
		statement.ExprColumns.Add(column, expression)
	}
	return statement
}

// ForUpdate generates "SELECT ... FOR UPDATE" statement
func (statement *Statement) ForUpdate() *Statement {
	statement.IsForUpdate = true
	return statement
}

// Nullable Update use only: update columns to null when value is nullable and zero-value
func (statement *Statement) Nullable(columns ...string) {
	newColumns := col2NewCols(columns...)
	for _, nc := range newColumns {
		statement.NullableMap[strings.ToLower(nc)] = true
	}
}

// Top generate LIMIT limit statement
func (statement *Statement) Top(limit int) *Statement {
	statement.Limit(limit)
	return statement
}

// Limit generate LIMIT start, limit statement
func (statement *Statement) Limit(limit int, start ...int) *Statement {
	statement.LimitN = &limit
	if len(start) > 0 {
		statement.Start = start[0]
	}
	return statement
}

// SetTable tempororily set table name, the parameter could be a string or a pointer of struct
func (statement *Statement) SetTable(tableNameOrBean interface{}) error {
	v := rValue(tableNameOrBean)
	t := v.Type()
	if t.Kind() == reflect.Struct {
		var err error
		statement.RefTable, err = statement.tagParser.ParseWithCache(v)
		if err != nil {
			return err
		}
	}

	statement.AltTableName = dialects.FullTableName(statement.dialect, statement.tagParser.GetTableMapper(), tableNameOrBean, true)
	return nil
}

// GroupBy generate "Group By keys" statement
func (statement *Statement) GroupBy(keys string) *Statement {
	statement.GroupByStr = statement.ReplaceQuote(keys)
	return statement
}

func (statement *Statement) WriteGroupBy(w builder.Writer) error {
	if statement.GroupByStr == "" {
		return nil
	}
	_, err := fmt.Fprintf(w, " GROUP BY %s", statement.GroupByStr)
	return err
}

// Having generate "Having conditions" statement
func (statement *Statement) Having(conditions string) *Statement {
	statement.HavingStr = fmt.Sprintf("HAVING %v", statement.ReplaceQuote(conditions))
	return statement
}

func (statement *Statement) writeHaving(w builder.Writer) error {
	if statement.HavingStr == "" {
		return nil
	}
	_, err := fmt.Fprint(w, " ", statement.HavingStr)
	return err
}

// SetUnscoped always disable struct tag "deleted"
func (statement *Statement) SetUnscoped() *Statement {
	statement.unscoped = true
	return statement
}

// GetUnscoped return true if it's unscoped
func (statement *Statement) GetUnscoped() bool {
	return statement.unscoped
}

// GenIndexSQL generated create index SQL
func (statement *Statement) GenIndexSQL() []string {
	var sqls []string
	tbName := statement.TableName()
	for _, index := range statement.RefTable.Indexes {
		if index.Type == schemas.IndexType {
			sql := statement.dialect.CreateIndexSQL(tbName, index)
			sqls = append(sqls, sql)
		}
	}
	return sqls
}

// GenUniqueSQL generates unique SQL
func (statement *Statement) GenUniqueSQL() []string {
	var sqls []string
	tbName := statement.TableName()
	for _, index := range statement.RefTable.Indexes {
		if index.Type == schemas.UniqueType {
			sql := statement.dialect.CreateIndexSQL(tbName, index)
			sqls = append(sqls, sql)
		}
	}
	return sqls
}

// GenDelIndexSQL generate delete index SQL
func (statement *Statement) GenDelIndexSQL() []string {
	var sqls []string
	tbName := statement.TableName()
	idx := strings.Index(tbName, ".")
	if idx > -1 {
		tbName = tbName[idx+1:]
	}
	for _, index := range statement.RefTable.Indexes {
		sqls = append(sqls, statement.dialect.DropIndexSQL(tbName, index))
	}
	return sqls
}

func (statement *Statement) asDBCond(fieldValue reflect.Value, fieldType reflect.Type, col *schemas.Column, allUseBool, requiredField bool) (interface{}, bool, error) {
	switch fieldType.Kind() {
	case reflect.Ptr:
		if fieldValue.IsNil() {
			return nil, true, nil
		}
		return statement.asDBCond(fieldValue.Elem(), fieldType.Elem(), col, allUseBool, requiredField)
	case reflect.Bool:
		if allUseBool || requiredField {
			return fieldValue.Interface(), true, nil
		}
		// if a bool in a struct, it will not be as a condition because it default is false,
		// please use Where() instead
		return nil, false, nil
	case reflect.String:
		if !requiredField && fieldValue.String() == "" {
			return nil, false, nil
		}
		// for MyString, should convert to string or panic
		if fieldType.String() != reflect.String.String() {
			return fieldValue.String(), true, nil
		}
		return fieldValue.Interface(), true, nil
	case reflect.Int8, reflect.Int16, reflect.Int, reflect.Int32, reflect.Int64:
		if !requiredField && fieldValue.Int() == 0 {
			return nil, false, nil
		}
		return fieldValue.Interface(), true, nil
	case reflect.Float32, reflect.Float64:
		if !requiredField && fieldValue.Float() == 0.0 {
			return nil, false, nil
		}
		return fieldValue.Interface(), true, nil
	case reflect.Uint8, reflect.Uint16, reflect.Uint, reflect.Uint32, reflect.Uint64:
		if !requiredField && fieldValue.Uint() == 0 {
			return nil, false, nil
		}
		return fieldValue.Interface(), true, nil
	case reflect.Struct:
		if fieldType.ConvertibleTo(schemas.TimeType) {
			t := fieldValue.Convert(schemas.TimeType).Interface().(time.Time)
			if !requiredField && (t.IsZero() || !fieldValue.IsValid()) {
				return nil, false, nil
			}
			res, err := dialects.FormatColumnTime(statement.dialect, statement.defaultTimeZone, col, t)
			if err != nil {
				return nil, false, err
			}
			return res, true, nil
		} else if fieldType.ConvertibleTo(schemas.BigFloatType) {
			t := fieldValue.Convert(schemas.BigFloatType).Interface().(big.Float)
			v := t.String()
			if v == "0" {
				return nil, false, nil
			}
			return t.String(), true, nil
		} else if _, ok := reflect.New(fieldType).Interface().(convert.Conversion); ok {
			return nil, false, nil
		} else if valNul, ok := fieldValue.Interface().(driver.Valuer); ok {
			val, _ := valNul.Value()
			if val == nil && !requiredField {
				return nil, false, nil
			}
			return val, true, nil
		} else {
			if col.IsJSON {
				if col.SQLType.IsText() {
					bytes, err := json.DefaultJSONHandler.Marshal(fieldValue.Interface())
					if err != nil {
						return nil, false, err
					}
					return string(bytes), true, nil
				} else if col.SQLType.IsBlob() {
					var bytes []byte
					var err error
					bytes, err = json.DefaultJSONHandler.Marshal(fieldValue.Interface())
					if err != nil {
						return nil, false, err
					}
					return bytes, true, nil
				}
			} else {
				table, err := statement.tagParser.ParseWithCache(fieldValue)
				if err != nil {
					return fieldValue.Interface(), true, nil
				}

				if len(table.PrimaryKeys) == 1 {
					pkField := reflect.Indirect(fieldValue).FieldByName(table.PKColumns()[0].FieldName)
					// fix non-int pk issues
					// if pkField.Int() != 0 {
					if pkField.IsValid() && !utils.IsZero(pkField.Interface()) {
						return pkField.Interface(), true, nil
					}
					return nil, false, nil
				}
				return nil, false, fmt.Errorf("not supported %v as %v", fieldValue.Interface(), table.PrimaryKeys)
			}
		}
	case reflect.Array:
		return nil, false, nil
	case reflect.Slice, reflect.Map:
		if fieldValue == reflect.Zero(fieldType) {
			return nil, false, nil
		}
		if fieldValue.IsNil() || !fieldValue.IsValid() || fieldValue.Len() == 0 {
			return nil, false, nil
		}

		if col.SQLType.IsText() {
			bytes, err := json.DefaultJSONHandler.Marshal(fieldValue.Interface())
			if err != nil {
				return nil, false, err
			}
			return string(bytes), true, nil
		} else if col.SQLType.IsBlob() {
			var bytes []byte
			var err error
			if (fieldType.Kind() == reflect.Array || fieldType.Kind() == reflect.Slice) &&
				fieldType.Elem().Kind() == reflect.Uint8 {
				if fieldValue.Len() > 0 {
					return fieldValue.Bytes(), true, nil
				}
				return nil, false, nil
			}
			bytes, err = json.DefaultJSONHandler.Marshal(fieldValue.Interface())
			if err != nil {
				return nil, false, err
			}
			return bytes, true, nil
		}
		return nil, false, nil
	}
	return fieldValue.Interface(), true, nil
}

func (statement *Statement) buildConds2(table *schemas.Table, bean interface{},
	includeVersion bool, includeUpdated bool, includeNil bool,
	includeAutoIncr bool, allUseBool bool, useAllCols bool, unscoped bool,
	mustColumnMap map[string]bool, tableName, aliasName string, addedTableName bool,
) (builder.Cond, error) {
	var conds []builder.Cond
	for _, col := range table.Columns() {
		if !includeVersion && col.IsVersion {
			continue
		}
		if !includeUpdated && col.IsUpdated {
			continue
		}
		if !includeAutoIncr && col.IsAutoIncrement {
			continue
		}

		if col.IsJSON {
			continue
		}

		var colName string
		if addedTableName {
			nm := tableName
			if len(aliasName) > 0 {
				nm = aliasName
			}
			colName = statement.quote(nm) + "." + statement.quote(col.Name)
		} else {
			colName = statement.quote(col.Name)
		}

		fieldValuePtr, err := col.ValueOf(bean)
		if err != nil {
			continue
		} else if fieldValuePtr == nil {
			continue
		}

		if col.IsDeleted && !unscoped { // tag "deleted" is enabled
			conds = append(conds, statement.CondDeleted(col))
		}

		fieldValue := *fieldValuePtr
		if fieldValue.Interface() == nil {
			continue
		}

		if statement.dialect.URI().DBType == schemas.MSSQL && (col.SQLType.Name == schemas.Text ||
			col.SQLType.IsBlob() || col.SQLType.Name == schemas.TimeStampz) {
			if utils.IsValueZero(fieldValue) {
				continue
			}

			return nil, fmt.Errorf("column %s is a TEXT type with data %#v which cannot be as compare condition", col.Name, fieldValue.Interface())
		}

		requiredField := useAllCols
		if b, ok := getFlagForColumn(mustColumnMap, col); ok {
			if b {
				requiredField = true
			} else {
				continue
			}
		}

		fieldType := reflect.TypeOf(fieldValue.Interface())
		if fieldType.Kind() == reflect.Ptr {
			if fieldValue.IsNil() {
				if includeNil {
					conds = append(conds, builder.Eq{colName: nil})
				}
				continue
			} else if !fieldValue.IsValid() {
				continue
			} else {
				// dereference ptr type to instance type
				fieldValue = fieldValue.Elem()
				fieldType = reflect.TypeOf(fieldValue.Interface())
				requiredField = true
			}
		}

		val, ok, err := statement.asDBCond(fieldValue, fieldType, col, allUseBool, requiredField)
		if err != nil {
			return nil, err
		}
		if !ok {
			continue
		}

		conds = append(conds, builder.Eq{colName: val})
	}

	return builder.And(conds...), nil
}

// BuildConds builds condition
func (statement *Statement) BuildConds(table *schemas.Table, bean interface{}, includeVersion bool, includeUpdated bool, includeNil bool, includeAutoIncr bool, addedTableName bool) (builder.Cond, error) {
	return statement.buildConds2(table, bean, includeVersion, includeUpdated, includeNil, includeAutoIncr, statement.allUseBool, statement.useAllCols,
		statement.unscoped, statement.MustColumnMap, statement.TableName(), statement.TableAlias, addedTableName)
}

// MergeConds merge conditions from bean and id
func (statement *Statement) MergeConds(bean interface{}) error {
	if !statement.NoAutoCondition && statement.RefTable != nil {
		addedTableName := (len(statement.JoinStr) > 0)
		autoCond, err := statement.BuildConds(statement.RefTable, bean, true, true, false, true, addedTableName)
		if err != nil {
			return err
		}
		statement.cond = statement.cond.And(autoCond)
	}

	return statement.ProcessIDParam()
}

func (statement *Statement) quoteColumnStr(columnStr string) string {
	columns := strings.Split(columnStr, ",")
	return statement.dialect.Quoter().Join(columns, ",")
}

// ConvertSQLOrArgs converts sql or args
func (statement *Statement) ConvertSQLOrArgs(sqlOrArgs ...interface{}) (string, []interface{}, error) {
	sql, args, err := statement.convertSQLOrArgs(sqlOrArgs...)
	if err != nil {
		return "", nil, err
	}
	return statement.ReplaceQuote(sql), args, nil
}

func (statement *Statement) convertSQLOrArgs(sqlOrArgs ...interface{}) (string, []interface{}, error) {
	switch sqlOrArgs[0].(type) {
	case string:
		if len(sqlOrArgs) > 1 {
			newArgs := make([]interface{}, 0, len(sqlOrArgs)-1)
			for _, arg := range sqlOrArgs[1:] {
				if v, ok := arg.(time.Time); ok {
					newArgs = append(newArgs, v.In(statement.defaultTimeZone).Format("2006-01-02 15:04:05"))
				} else if v, ok := arg.(*time.Time); ok && v != nil {
					newArgs = append(newArgs, v.In(statement.defaultTimeZone).Format("2006-01-02 15:04:05"))
				} else {
					newArgs = append(newArgs, arg)
				}
			}
			return sqlOrArgs[0].(string), newArgs, nil
		}
		return sqlOrArgs[0].(string), sqlOrArgs[1:], nil
	case *builder.Builder:
		return sqlOrArgs[0].(*builder.Builder).ToSQL()
	case builder.Builder:
		bd := sqlOrArgs[0].(builder.Builder)
		return bd.ToSQL()
	}

	return "", nil, ErrUnSupportedType
}

func (statement *Statement) joinColumns(cols []*schemas.Column, includeTableName bool) string {
	colnames := make([]string, len(cols))
	for i, col := range cols {
		if includeTableName {
			colnames[i] = statement.quote(statement.TableName()) +
				"." + statement.quote(col.Name)
		} else {
			colnames[i] = statement.quote(col.Name)
		}
	}
	return strings.Join(colnames, ", ")
}

// CondDeleted returns the conditions whether a record is soft deleted.
func (statement *Statement) CondDeleted(col *schemas.Column) builder.Cond {
	colName := statement.quote(col.Name)
	if statement.JoinStr != "" {
		var prefix string
		if statement.TableAlias != "" {
			prefix = statement.TableAlias
		} else {
			prefix = statement.TableName()
		}
		colName = statement.quote(prefix) + "." + statement.quote(col.Name)
	}
	cond := builder.NewCond()
	if col.SQLType.IsNumeric() {
		cond = builder.Eq{colName: 0}
	} else {
		// FIXME: mssql: The conversion of a nvarchar data type to a datetime data type resulted in an out-of-range value.
		if statement.dialect.URI().DBType != schemas.MSSQL {
			cond = builder.Eq{colName: utils.ZeroTime1}
		}
	}

	if col.Nullable {
		cond = cond.Or(builder.IsNull{colName})
	}

	return cond
}

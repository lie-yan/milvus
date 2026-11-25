package schema

import (
	"testing"

	"github.com/stretchr/testify/require"
	"google.golang.org/protobuf/proto"

	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
	"github.com/milvus-io/milvus/pkg/v3/proto/planpb"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
	"github.com/milvus-io/milvus/pkg/v3/util/typeutil"
)

func schemaFixture(t *testing.T) (*schemapb.CollectionSchema, *typeutil.SchemaHelper, *View) {
	t.Helper()
	schema := &schemapb.CollectionSchema{
		Fields: []*schemapb.FieldSchema{
			{FieldID: 101, Name: "i", DataType: schemapb.DataType_Int64, IsPrimaryKey: true, AutoID: true, IsPartitionKey: true, IsClusteringKey: true},
			{FieldID: 102, Name: "b", DataType: schemapb.DataType_Bool},
			{FieldID: 103, Name: "nullable", DataType: schemapb.DataType_Int64, Nullable: true},
			{FieldID: 104, Name: "narrow", DataType: schemapb.DataType_Int8},
			{FieldID: 105, Name: "json", DataType: schemapb.DataType_JSON},
		},
		StructArrayFields: []*schemapb.StructArrayFieldSchema{{FieldID: 200, Name: "items", Fields: []*schemapb.FieldSchema{
			{FieldID: 201, Name: "sub", DataType: schemapb.DataType_Int64},
		}}},
	}
	helper, err := typeutil.CreateSchemaHelper(schema)
	require.NoError(t, err)
	view, err := NewView(helper, helper.TopLevelFields())
	require.NoError(t, err)
	return schema, helper, view
}

func columnFixture(field *schemapb.FieldSchema) *planpb.ColumnInfo {
	return &planpb.ColumnInfo{
		FieldId: field.GetFieldID(), DataType: field.GetDataType(), Nullable: field.GetNullable(),
		ElementType: field.GetElementType(), IsPrimaryKey: field.GetIsPrimaryKey(), IsAutoID: field.GetAutoID(),
		IsPartitionKey: field.GetIsPartitionKey(), IsClusteringKey: field.GetIsClusteringKey(),
	}
}

func TestSchemaViewColumnScope(t *testing.T) {
	schema, helper, view := schemaFixture(t)
	for _, field := range schema.GetFields() {
		info := columnFixture(field)
		resolved, supported, err := view.CheckColumn(info)
		require.NoError(t, err)
		require.True(t, supported, field.GetName())
		require.Same(t, field, resolved)
	}
	sub, err := helper.GetFieldFromID(201)
	require.NoError(t, err) // Successful helper lookup does not establish document scope.
	for name, info := range map[string]*planpb.ColumnInfo{
		"flattened subfield": columnFixture(sub),
		"path":               {FieldId: 105, DataType: schemapb.DataType_JSON, NestedPath: []string{"key"}},
		"element":            {FieldId: 201, DataType: schemapb.DataType_Int64, IsElementLevel: true},
	} {
		t.Run(name, func(t *testing.T) {
			resolved, supported, err := view.CheckColumn(info)
			require.NoError(t, err)
			require.False(t, supported)
			require.Nil(t, resolved)
		})
	}
	info := columnFixture(schema.Fields[0])
	info.ProtoReflect().SetUnknown([]byte{0x98, 0x06, 0x01})
	_, supported, err := view.CheckColumn(info)
	require.NoError(t, err)
	require.False(t, supported)
}

func TestSchemaViewMetadataConsistency(t *testing.T) {
	schema, _, view := schemaFixture(t)
	for name, change := range map[string]func(*planpb.ColumnInfo){
		"datatype":       func(c *planpb.ColumnInfo) { c.DataType = schemapb.DataType_Bool },
		"nullability":    func(c *planpb.ColumnInfo) { c.Nullable = true },
		"element type":   func(c *planpb.ColumnInfo) { c.ElementType = schemapb.DataType_Int64 },
		"primary key":    func(c *planpb.ColumnInfo) { c.IsPrimaryKey = false },
		"auto ID":        func(c *planpb.ColumnInfo) { c.IsAutoID = false },
		"partition key":  func(c *planpb.ColumnInfo) { c.IsPartitionKey = false },
		"clustering key": func(c *planpb.ColumnInfo) { c.IsClusteringKey = false },
		"missing field":  func(c *planpb.ColumnInfo) { c.FieldId = 999 },
	} {
		t.Run(name, func(t *testing.T) {
			info := columnFixture(schema.Fields[0])
			change(info)
			field, supported, err := view.CheckColumn(info)
			require.ErrorIs(t, err, merr.ErrServiceInternal)
			require.False(t, supported)
			require.Nil(t, field)
		})
	}
}

func TestSchemaViewPinnedMembership(t *testing.T) {
	schema, helper, view := schemaFixture(t)
	// Simulate entry replacement without mutating live fields or exercising the proxy cache.
	replacement := proto.Clone(schema).(*schemapb.CollectionSchema)
	replacement.Fields[0].DataType = schemapb.DataType_Bool
	nextHelper, err := typeutil.CreateSchemaHelper(replacement)
	require.NoError(t, err)
	_, err = NewView(nextHelper, schema.GetFields())
	require.ErrorIs(t, err, merr.ErrServiceInternal)
	nextView, err := NewView(nextHelper, nextHelper.TopLevelFields())
	require.NoError(t, err)
	info := columnFixture(schema.Fields[0])
	field, supported, err := view.CheckColumn(info)
	require.NoError(t, err)
	require.True(t, supported)
	require.Same(t, schema.Fields[0], field)
	_, _, err = nextView.CheckColumn(info)
	require.ErrorIs(t, err, merr.ErrServiceInternal)
	// Membership is copied, while the immutable field nodes remain borrowed.
	membership := append([]*schemapb.FieldSchema(nil), schema.Fields...)
	copied, err := NewView(helper, membership)
	require.NoError(t, err)
	membership[0] = replacement.Fields[0]
	_, supported, err = copied.CheckColumn(info)
	require.NoError(t, err)
	require.True(t, supported)
}

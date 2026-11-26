package translation_test

import (
	"context"
	"math"
	"testing"

	"github.com/stretchr/testify/require"
	"google.golang.org/protobuf/proto"

	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
	adapterruntime "github.com/milvus-io/milvus/internal/parser/planparserv2/predicateadapter/runtime"
	"github.com/milvus-io/milvus/internal/parser/planparserv2/predicateadapter/schema"
	"github.com/milvus-io/milvus/internal/parser/planparserv2/predicateadapter/translation"
	"github.com/milvus-io/milvus/pkg/v3/proto/planpb"
	"github.com/milvus-io/milvus/pkg/v3/util/typeutil"
)

// Canonical target fixtures establish semantics before E02/E03 change emitted structure.
func TestSemanticExportContract(t *testing.T) {
	translator, source := compositionFixture(t)
	parts := compositionParts(source)
	parts[1].IsTemplate = true
	parts[1].GetUnaryRangeExpr().TemplateVariableName = "threshold"
	original := proto.Clone(source)
	targetParts := compositionParts(proto.Clone(source).(*planpb.Expr))
	targetParts[1].IsTemplate = false
	targetParts[1].GetUnaryRangeExpr().TemplateVariableName = ""
	field := &planpb.ColumnInfo{FieldId: 104, DataType: parts[3].GetBinaryRangeExpr().ColumnInfo.DataType}
	targetParts[3] = binarySource(planpb.BinaryExpr_LogicalAnd,
		unarySource(field, planpb.OpType_GreaterEqual, floatSource(0.5)),
		unarySource(field, planpb.OpType_LessEqual, floatSource(2)))
	targetParts[4] = &planpb.Expr{Expr: &planpb.Expr_TermExpr{TermExpr: &planpb.TermExpr{
		ColumnInfo: proto.Clone(parts[4].GetTermExpr().ColumnInfo).(*planpb.ColumnInfo),
		Values:     []*planpb.GenericValue{stringSource("a"), stringSource("z"), stringSource("z")},
	}}}
	// Left association also differs from the source's right association.
	target := targetParts[0]
	for _, part := range targetParts[1:] {
		target = binarySource(planpb.BinaryExpr_LogicalAnd, target, part)
	}
	imported := importSource(t, translator, source)
	output, err := translator.Export(context.Background(), imported, imported.Root())
	require.NoError(t, err)
	for _, tc := range []struct {
		name  string
		field int64
		value *planpb.GenericValue
		want  bool
	}{
		{"matching", 101, integerSource(7), true},
		{"below threshold", 101, integerSource(6), false},
		{"equal columns", 103, integerSource(1), false},
		{"lower bound", 104, floatSource(0.5), true},
		{"upper bound", 104, floatSource(2), true},
		{"outside range", 104, floatSource(2.5), false},
		{"sorted member", 105, stringSource("a"), true},
		{"missing member", 105, stringSource("b"), false},
		{"nonnull opaque", 106, floatSource(1), false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			row := map[int64]*planpb.GenericValue{101: integerSource(7), 102: integerSource(1), 103: integerSource(2), 104: floatSource(1), 105: stringSource("z"), 106: nil}
			row[tc.field] = tc.value
			for name, expr := range map[string]*planpb.Expr{"source": source, "target fixture": target, "current export": output} {
				require.Equal(t, tc.want, evaluatePredicate(t, expr, row), name)
			}
		})
	}
	require.True(t, proto.Equal(original, source), "source remains caller-owned")
}

// Explicit truth values guard integer precision, floating zero and Boolean semantics.
func TestSemanticOracleScalarBoundaries(t *testing.T) {
	for _, tc := range []struct {
		left, right *planpb.GenericValue
		op          planpb.OpType
		want        bool
	}{
		{integerSource(9007199254740993), integerSource(9007199254740992), planpb.OpType_GreaterThan, true},
		{integerSource(math.MinInt64), integerSource(math.MaxInt64), planpb.OpType_LessThan, true},
		{floatSource(math.Copysign(0, -1)), floatSource(0), planpb.OpType_Equal, true},
		{stringSource("a"), stringSource("z"), planpb.OpType_Equal, false},
		{&planpb.GenericValue{Val: &planpb.GenericValue_BoolVal{BoolVal: true}}, &planpb.GenericValue{Val: &planpb.GenericValue_BoolVal{}}, planpb.OpType_NotEqual, true},
	} {
		require.Equal(t, tc.want, compareValues(t, tc.op, tc.left, tc.right))
	}
}

// Expected executable metadata comes from the schema, independent of template provenance.
func TestSemanticExportFieldMetadata(t *testing.T) {
	for _, field := range []*schemapb.FieldSchema{
		{FieldID: 101, Name: "primary", DataType: schemapb.DataType_Int64, IsPrimaryKey: true, AutoID: true},
		{FieldID: 101, Name: "partition", DataType: schemapb.DataType_Int64, IsPartitionKey: true},
		{FieldID: 101, Name: "clustering", DataType: schemapb.DataType_Int64, IsClusteringKey: true},
	} {
		t.Run(field.Name, func(t *testing.T) {
			helper, err := typeutil.CreateSchemaHelper(&schemapb.CollectionSchema{Fields: []*schemapb.FieldSchema{field}})
			require.NoError(t, err)
			view, err := schema.NewView(helper, []*schemapb.FieldSchema{field})
			require.NoError(t, err)
			runtime, err := adapterruntime.SharedRuntime()
			require.NoError(t, err)
			translator, err := translation.New(runtime, view, runtime.Types())
			require.NoError(t, err)
			expected := &planpb.ColumnInfo{FieldId: 101, DataType: schemapb.DataType_Int64,
				IsPrimaryKey: field.IsPrimaryKey, IsAutoID: field.AutoID, IsPartitionKey: field.IsPartitionKey, IsClusteringKey: field.IsClusteringKey}
			source := unarySource(proto.Clone(expected).(*planpb.ColumnInfo), planpb.OpType_Equal, integerSource(7))
			source.IsTemplate = true
			source.GetUnaryRangeExpr().TemplateVariableName = "bound"
			target := unarySource(expected, planpb.OpType_Equal, integerSource(7))
			imported := importSource(t, translator, source)
			output, err := translator.Export(context.Background(), imported, imported.Root())
			require.NoError(t, err)
			require.True(t, proto.Equal(expected, output.GetUnaryRangeExpr().ColumnInfo))
			for _, value := range []int64{6, 7, 8} {
				row := map[int64]*planpb.GenericValue{101: integerSource(value)}
				require.Equal(t, value == 7, evaluatePredicate(t, target, row))
				require.Equal(t, value == 7, evaluatePredicate(t, output, row))
			}
			require.True(t, proto.Equal(target, output), "fully bound native output")
		})
	}
}

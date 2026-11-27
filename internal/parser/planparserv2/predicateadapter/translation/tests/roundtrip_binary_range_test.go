package translation_test

import (
	"context"
	"math"
	"testing"

	"github.com/lie-yan/predicate-optimization/expressions"
	"github.com/lie-yan/predicate-optimization/foundation"
	"github.com/stretchr/testify/require"
	"google.golang.org/protobuf/proto"

	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
	"github.com/milvus-io/milvus/internal/parser/planparserv2/predicateadapter/translation"
	"github.com/milvus-io/milvus/pkg/v3/proto/planpb"
)

func floatSource(value float64) *planpb.GenericValue {
	return &planpb.GenericValue{Val: &planpb.GenericValue_FloatVal{FloatVal: value}}
}

func rangeSource(column *planpb.ColumnInfo, lower, upper *planpb.GenericValue) *planpb.Expr {
	return &planpb.Expr{IsTemplate: true, Expr: &planpb.Expr_BinaryRangeExpr{BinaryRangeExpr: &planpb.BinaryRangeExpr{
		ColumnInfo: column, LowerValue: lower, UpperValue: upper, LowerInclusive: true, UpperInclusive: true,
		LowerTemplateVariableName: "lower", UpperTemplateVariableName: "upper",
	}}}
}

func TestBinaryRangeMappings(t *testing.T) {
	for _, tc := range []struct {
		typ          schemapb.DataType
		lower, upper *planpb.GenericValue
	}{
		{schemapb.DataType_Int8, integerSource(-128), integerSource(127)},
		{schemapb.DataType_Int16, integerSource(-32768), integerSource(32767)},
		{schemapb.DataType_Int32, integerSource(math.MinInt32), integerSource(math.MaxInt32)},
		{schemapb.DataType_Int64, integerSource(math.MinInt64), integerSource(math.MaxInt64)},
		{schemapb.DataType_Float, floatSource(-math.MaxFloat32), floatSource(math.MaxFloat32)},
		{schemapb.DataType_Double, floatSource(-math.MaxFloat64), floatSource(math.MaxFloat64)},
	} {
		for _, nullable := range []bool{false, true} {
			t.Run(tc.typ.String(), func(t *testing.T) {
				translator, null, _ := opaqueFixture(t, tc.typ, nullable)
				for _, lowerInc := range []bool{false, true} {
					for _, upperInc := range []bool{false, true} {
						source := rangeSource(null.GetNullExpr().ColumnInfo, tc.lower, tc.upper)
						source.GetBinaryRangeExpr().LowerInclusive = lowerInc
						source.GetBinaryRangeExpr().UpperInclusive = upperInc
						imported := importSource(t, translator, source)
						graph := imported.Root()
						require.True(t, graph.ChildAt(0).ChildAt(0).(*expressions.ReferenceExpression).ID().EqualsTo(graph.ChildAt(1).ChildAt(0).(*expressions.ReferenceExpression).ID()))
						rebuilt := logicalGraph(t, foundation.And, rebuiltCall(t, graph.ChildAt(0).(*expressions.CallExpression), graph.ChildAt(0).ChildAt(0), graph.ChildAt(0).ChildAt(1)), graph.ChildAt(1))
						output, err := translator.Export(context.Background(), imported, rebuilt)
						require.NoError(t, err)
						require.True(t, proto.Equal(expectedExport(source), output))
					}
				}
			})
		}
	}
	// Equal endpoints preserve separate roles and signed-zero payloads.
	translator, null, _ := opaqueFixture(t, schemapb.DataType_Double, false)
	source := rangeSource(null.GetNullExpr().ColumnInfo, floatSource(math.Copysign(0, -1)), floatSource(0))
	imported := importSource(t, translator, source)
	output, err := translator.Export(context.Background(), imported, imported.Root())
	require.NoError(t, err)
	require.True(t, math.Signbit(output.GetBinaryExpr().Left.GetUnaryRangeExpr().Value.GetFloatVal()))
	require.False(t, math.Signbit(output.GetBinaryExpr().Right.GetUnaryRangeExpr().Value.GetFloatVal()))
}

func TestBinaryRangeImportCoverage(t *testing.T) {
	for _, tc := range []struct {
		typ          schemapb.DataType
		lower, upper *planpb.GenericValue
		kind         translation.FailureKind
	}{
		{schemapb.DataType_Int8, integerSource(-129), integerSource(127), translation.UnsupportedExpression},
		{schemapb.DataType_Int8, integerSource(0), integerSource(300), translation.UnsupportedExpression},
		{schemapb.DataType_Float, floatSource(0), floatSource(0.1), translation.UnsupportedExpression},
		{schemapb.DataType_Double, integerSource(0), floatSource(1), translation.UnsupportedExpression},
		{schemapb.DataType_Double, floatSource(0), floatSource(math.NaN()), translation.UnsupportedExpression},
		{schemapb.DataType_Int64, integerSource(2), integerSource(1), translation.UnsafeSource},
		{schemapb.DataType_Int64, integerSource(0), floatSource(1), translation.UnsafeSource},
	} {
		translator, null, _ := opaqueFixture(t, tc.typ, false)
		source := rangeSource(null.GetNullExpr().ColumnInfo, tc.lower, tc.upper)
		imported, err := translator.Import(context.Background(), source)
		require.Nil(t, imported)
		assertFailure(t, err, tc.kind, translation.StageImport)
	}
	for _, tc := range []struct {
		typ      schemapb.DataType
		nullable bool
	}{
		{schemapb.DataType_Bool, false}, {schemapb.DataType_VarChar, false},
	} {
		translator, fields := compareFixture(t, tc.typ, tc.typ, tc.nullable)
		source := rangeSource(fields.GetCompareExpr().RightColumnInfo, integerSource(0), integerSource(1))
		_, err := translator.Import(context.Background(), source)
		assertFailure(t, err, translation.UnsupportedExpression, translation.StageImport)
	}
	translator, null, _ := opaqueFixture(t, schemapb.DataType_Int64, false)
	for _, change := range []func(*planpb.BinaryRangeExpr){
		func(n *planpb.BinaryRangeExpr) { n.UpperValue = nil },
		func(n *planpb.BinaryRangeExpr) { n.ColumnInfo.FieldId = 999 },
		func(n *planpb.BinaryRangeExpr) { n.LowerInclusive = false; n.UpperValue = integerSource(0) },
	} {
		source := rangeSource(proto.Clone(null.GetNullExpr().ColumnInfo).(*planpb.ColumnInfo), integerSource(0), integerSource(1))
		change(source.GetBinaryRangeExpr())
		_, err := translator.Import(context.Background(), source)
		assertFailure(t, err, translation.UnsafeSource, translation.StageImport)
	}
}

func TestBinaryRangeRolesAndOwnership(t *testing.T) {
	ctx := context.Background()
	translator, null, _ := opaqueFixture(t, schemapb.DataType_Int64, false)
	field := null.GetNullExpr().ColumnInfo
	source := rangeSource(field, integerSource(7), integerSource(7))
	conjunction := binarySource(planpb.BinaryExpr_LogicalAnd, unarySource(field, planpb.OpType_GreaterEqual, integerSource(7)), unarySource(field, planpb.OpType_LessEqual, integerSource(7)))
	mixed := binarySource(planpb.BinaryExpr_LogicalOr, source, conjunction)
	imported := importSource(t, translator, mixed)
	root := imported.Root()
	require.True(t, root.ChildAt(0).EqualsTo(root.ChildAt(1)))
	// Equal graphs export equivalent conjunctions regardless of source syntax.
	shared := logicalGraph(t, foundation.Or, root.ChildAt(0), root.ChildAt(0))
	output, err := translator.Export(ctx, imported, shared)
	require.NoError(t, err)
	require.True(t, proto.Equal(expectedExport(mixed), output))
	output.GetBinaryExpr().Left.GetBinaryExpr().Left.GetUnaryRangeExpr().Value.Val.(*planpb.GenericValue_Int64Val).Int64Val = 99
	require.Equal(t, int64(7), output.GetBinaryExpr().Left.GetBinaryExpr().Right.GetUnaryRangeExpr().Value.GetInt64Val())
	require.Equal(t, int64(7), output.GetBinaryExpr().Right.GetBinaryExpr().Left.GetUnaryRangeExpr().Value.GetInt64Val())
	again, err := translator.Export(ctx, imported, shared)
	require.NoError(t, err)
	require.True(t, proto.Equal(expectedExport(mixed), again))
}

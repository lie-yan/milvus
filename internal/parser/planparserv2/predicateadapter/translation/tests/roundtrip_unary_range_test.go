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
	adapterruntime "github.com/milvus-io/milvus/internal/parser/planparserv2/predicateadapter/runtime"
	"github.com/milvus-io/milvus/internal/parser/planparserv2/predicateadapter/translation"
	"github.com/milvus-io/milvus/pkg/v3/proto/planpb"
)

func TestUnaryComparisonMappings(t *testing.T) {
	ctx := context.Background()
	for _, tc := range []struct {
		typ    schemapb.DataType
		values []*planpb.GenericValue
	}{
		{schemapb.DataType_Bool, []*planpb.GenericValue{{Val: &planpb.GenericValue_BoolVal{}}, {Val: &planpb.GenericValue_BoolVal{BoolVal: true}}}},
		{schemapb.DataType_Int8, []*planpb.GenericValue{integerSource(-128), integerSource(127)}},
		{schemapb.DataType_Int16, []*planpb.GenericValue{integerSource(-32768), integerSource(32767)}},
		{schemapb.DataType_Int32, []*planpb.GenericValue{integerSource(math.MinInt32), integerSource(math.MaxInt32)}},
		{schemapb.DataType_Int64, []*planpb.GenericValue{integerSource(math.MinInt64), integerSource(math.MaxInt64), integerSource(9007199254740993), integerSource(0)}},
		{schemapb.DataType_Float, []*planpb.GenericValue{floatSource(-math.MaxFloat32), floatSource(1.5)}},
		{schemapb.DataType_Double, []*planpb.GenericValue{floatSource(-math.MaxFloat64), floatSource(math.Inf(1))}},
	} {
		for _, nullable := range []bool{false, true} {
			translator, null, _ := opaqueFixture(t, tc.typ, nullable)
			ops := []planpb.OpType{planpb.OpType_Equal, planpb.OpType_NotEqual}
			if tc.typ != schemapb.DataType_Bool {
				ops = append(ops, planpb.OpType_LessThan, planpb.OpType_LessEqual, planpb.OpType_GreaterThan, planpb.OpType_GreaterEqual)
			}
			for _, op := range ops {
				for _, value := range tc.values {
					source := unarySource(null.GetNullExpr().ColumnInfo, op, value)
					source.IsTemplate = true
					source.GetUnaryRangeExpr().TemplateVariableName = "filled"
					imported := importSource(t, translator, source)
					call := imported.Root().(*expressions.CallExpression)
					require.Equal(t, call.ChildAt(0).Type().Nullability(), call.Type().Nullability())
					require.Equal(t, foundation.NonNull, call.ChildAt(1).Type().Nullability())
					runtime, err := adapterruntime.SharedRuntime()
					require.NoError(t, err)
					recovered, err := runtime.ComparisonOperator(call.Handle())
					require.NoError(t, err)
					require.Equal(t, op, recovered)
					require.IsType(t, &expressions.ReferenceExpression{}, call.ChildAt(0))
					require.IsType(t, &expressions.LiteralExpression{}, call.ChildAt(1))
					rebuilt := rebuiltCall(t, call, call.ChildAt(0), call.ChildAt(1))
					output, err := translator.Export(ctx, imported, rebuilt)
					require.NoError(t, err)
					require.True(t, proto.Equal(expectedExport(source), output))
				}
			}
		}
	}
}

func TestUnaryComparisonCompositionAndOwnership(t *testing.T) {
	ctx := context.Background()
	translator, null, _ := opaqueFixture(t, schemapb.DataType_Int64, false)
	first := unarySource(null.GetNullExpr().ColumnInfo, planpb.OpType_GreaterThan, integerSource(7))
	second := proto.Clone(first).(*planpb.Expr)
	first.IsTemplate = true
	first.GetUnaryRangeExpr().TemplateVariableName = "first"
	second.GetUnaryRangeExpr().TemplateVariableName = "second"
	source := binarySource(planpb.BinaryExpr_LogicalAnd, first, binarySource(planpb.BinaryExpr_LogicalOr, second, notSource(null)))
	imported := importSource(t, translator, source)
	root := imported.Root().(*expressions.LogicalExpression)
	call := root.ChildAt(0).(*expressions.CallExpression)
	other := root.ChildAt(1).ChildAt(0).(*expressions.CallExpression)
	require.Same(t, call.ChildAt(0), other.ChildAt(0))
	// Shared graph nodes still produce independent executable output.
	rebuilt := logicalGraph(t, foundation.And, call, logicalGraph(t, foundation.Or, call, root.ChildAt(1).ChildAt(1)))
	output, err := translator.Export(ctx, imported, rebuilt)
	require.NoError(t, err)
	require.True(t, proto.Equal(expectedExport(source), output))
	another, err := translator.Export(ctx, imported, rebuilt)
	require.NoError(t, err)
	left := output.GetBinaryExpr().Left.GetUnaryRangeExpr()
	right := output.GetBinaryExpr().Right.GetBinaryExpr().Left.GetUnaryRangeExpr()
	require.NotSame(t, left.ColumnInfo, right.ColumnInfo)
	require.NotSame(t, left.Value.Val, right.Value.Val)
	left.Value.Val.(*planpb.GenericValue_Int64Val).Int64Val = 99
	left.ColumnInfo.NestedPath = []string{"changed"}
	require.Equal(t, int64(7), right.Value.GetInt64Val())
	require.Empty(t, right.ColumnInfo.NestedPath)
	require.True(t, proto.Equal(expectedExport(source), another))
}

func TestUnaryComparisonImportCoverage(t *testing.T) {
	for _, tc := range []struct {
		name   string
		mutate func(*planpb.Expr)
		kind   translation.FailureKind
	}{
		{"missing field", func(e *planpb.Expr) { e.GetUnaryRangeExpr().ColumnInfo = nil }, translation.UnsafeSource},
		{"missing value", func(e *planpb.Expr) { e.GetUnaryRangeExpr().Value = nil }, translation.UnsafeSource},
		{"wrong arm", func(e *planpb.Expr) { e.GetUnaryRangeExpr().Value.Val = &planpb.GenericValue_BoolVal{} }, translation.UnsafeSource},
		{"schema mismatch", func(e *planpb.Expr) { e.GetUnaryRangeExpr().ColumnInfo.Nullable = true }, translation.UnsafeSource},
		{"text match with options", func(e *planpb.Expr) {
			e.GetUnaryRangeExpr().Op = planpb.OpType_TextMatch
			e.GetUnaryRangeExpr().ExtraValues = []*planpb.GenericValue{integerSource(1)}
		}, translation.UnsupportedExpression},
		{"future operator", func(e *planpb.Expr) { e.GetUnaryRangeExpr().Op = 999 }, translation.UnsupportedExpression},
		{"unknown literal", func(e *planpb.Expr) { e.GetUnaryRangeExpr().Value.ProtoReflect().SetUnknown([]byte{0xa0, 0x06, 1}) }, translation.UnsupportedExpression},
	} {
		t.Run(tc.name, func(t *testing.T) {
			translator, null, _ := opaqueFixture(t, schemapb.DataType_Int64, false)
			source := unarySource(null.GetNullExpr().ColumnInfo, planpb.OpType_Equal, integerSource(0))
			tc.mutate(source)
			// A supported sibling must not escape admission as a partial result.
			imported, err := translator.Import(context.Background(), binarySource(planpb.BinaryExpr_LogicalAnd, trueSource(), source))
			require.Nil(t, imported)
			assertFailure(t, err, tc.kind, translation.StageImport)
		})
	}
	for _, tc := range []struct {
		typ      schemapb.DataType
		nullable bool
		op       planpb.OpType
		value    *planpb.GenericValue
	}{
		{schemapb.DataType_Int8, false, planpb.OpType_GreaterThan, integerSource(300)},
		{schemapb.DataType_Bool, false, planpb.OpType_LessThan, &planpb.GenericValue{Val: &planpb.GenericValue_BoolVal{}}},
	} {
		translator, null, _ := opaqueFixture(t, tc.typ, tc.nullable)
		imported, err := translator.Import(context.Background(), unarySource(null.GetNullExpr().ColumnInfo, tc.op, tc.value))
		require.Nil(t, imported)
		assertFailure(t, err, translation.UnsupportedExpression, translation.StageImport)
	}
}

func TestUnaryComparisonGraphAndTranslationDefects(t *testing.T) {
	ctx := context.Background()
	translator, null, _ := opaqueFixture(t, schemapb.DataType_Int64, false)
	source := unarySource(null.GetNullExpr().ColumnInfo, planpb.OpType_GreaterThan, integerSource(7))
	imported := importSource(t, translator, source)
	call := imported.Root().(*expressions.CallExpression)
	otherImport, err := translator.Import(ctx, source)
	require.NoError(t, err)
	foreignReference := otherImport.Root().ChildAt(0)
	require.NotSame(t, call.ChildAt(0), foreignReference)
	for _, graph := range []expressions.Expression{
		rebuiltCall(t, call, foreignReference, call.ChildAt(1)),
	} {
		output, err := translator.Export(ctx, imported, graph)
		require.Nil(t, output)
		assertFailure(t, err, translation.TranslationDefect, translation.StageExport)
	}
}

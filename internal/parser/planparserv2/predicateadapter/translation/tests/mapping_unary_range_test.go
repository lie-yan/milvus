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
	for _, typ := range []schemapb.DataType{schemapb.DataType_Int64, schemapb.DataType_Bool} {
		translator, null, _ := opaqueFixture(t, typ, false)
		ops := []planpb.OpType{planpb.OpType_Equal, planpb.OpType_NotEqual}
		values := []*planpb.GenericValue{{Val: &planpb.GenericValue_BoolVal{}}, {Val: &planpb.GenericValue_BoolVal{BoolVal: true}}}
		if typ == schemapb.DataType_Int64 {
			ops = append(ops, planpb.OpType_LessThan, planpb.OpType_LessEqual, planpb.OpType_GreaterThan, planpb.OpType_GreaterEqual)
			values = []*planpb.GenericValue{integerSource(math.MinInt64), integerSource(math.MaxInt64), integerSource(9007199254740993), integerSource(0)}
		}
		for _, op := range ops {
			for _, value := range values {
				source := unarySource(null.GetNullExpr().ColumnInfo, op, value)
				source.IsTemplate = true
				source.GetUnaryRangeExpr().TemplateVariableName = "filled"
				_, imported := importSource(t, translator, source)
				call := imported.Root().(*expressions.CallExpression)
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
				require.NoError(t, translator.ValidateCandidate(ctx, imported, output))
				require.True(t, proto.Equal(source, output))
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
	_, imported := importSource(t, translator, source)
	root := imported.Root().(*expressions.LogicalExpression)
	call := root.ChildAt(0).(*expressions.CallExpression)
	other := root.ChildAt(1).ChildAt(0).(*expressions.CallExpression)
	require.True(t, call.ChildAt(0).(*expressions.ReferenceExpression).ID().EqualsTo(other.ChildAt(0).(*expressions.ReferenceExpression).ID()))
	// Equal graph objects can be shared without merging occurrence-specific metadata.
	rebuilt := logicalGraph(t, foundation.And, call, logicalGraph(t, foundation.Or, call, root.ChildAt(1).ChildAt(1)))
	output, err := translator.Export(ctx, imported, rebuilt)
	require.NoError(t, err)
	require.NoError(t, translator.ValidateCandidate(ctx, imported, output))
	require.True(t, proto.Equal(source, output))
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
	require.True(t, proto.Equal(source, another))
	assertFailure(t, translator.ValidateCandidate(ctx, imported, output), translation.TranslationDefect, translation.StageCandidateValidation)
}

func TestUnaryComparisonAdmission(t *testing.T) {
	for _, tc := range []struct {
		name   string
		mutate func(*planpb.Expr)
		kind   translation.FailureKind
	}{
		{"missing node", func(e *planpb.Expr) { e.Expr = &planpb.Expr_UnaryRangeExpr{} }, translation.UnsafeSource},
		{"missing field", func(e *planpb.Expr) { e.GetUnaryRangeExpr().ColumnInfo = nil }, translation.UnsafeSource},
		{"missing value", func(e *planpb.Expr) { e.GetUnaryRangeExpr().Value = nil }, translation.UnsafeSource},
		{"wrong arm", func(e *planpb.Expr) { e.GetUnaryRangeExpr().Value.Val = &planpb.GenericValue_BoolVal{} }, translation.UnsafeSource},
		{"invalid operator", func(e *planpb.Expr) { e.GetUnaryRangeExpr().Op = planpb.OpType_Invalid }, translation.UnsafeSource},
		{"schema mismatch", func(e *planpb.Expr) { e.GetUnaryRangeExpr().ColumnInfo.Nullable = true }, translation.UnsafeSource},
		{"future operator", func(e *planpb.Expr) { e.GetUnaryRangeExpr().Op = 999 }, translation.UnsupportedSource},
		{"extra values", func(e *planpb.Expr) { e.GetUnaryRangeExpr().ExtraValues = []*planpb.GenericValue{integerSource(1)} }, translation.UnsupportedSource},
		{"unknown message", func(e *planpb.Expr) { e.GetUnaryRangeExpr().ProtoReflect().SetUnknown([]byte{0xa0, 0x06, 1}) }, translation.UnsupportedSource},
		{"unknown literal", func(e *planpb.Expr) { e.GetUnaryRangeExpr().Value.ProtoReflect().SetUnknown([]byte{0xa0, 0x06, 1}) }, translation.UnsupportedSource},
	} {
		t.Run(tc.name, func(t *testing.T) {
			translator, null, _ := opaqueFixture(t, schemapb.DataType_Int64, false)
			source := unarySource(null.GetNullExpr().ColumnInfo, planpb.OpType_Equal, integerSource(0))
			tc.mutate(source)
			// A supported sibling must not escape admission as a partial result.
			admitted, err := translator.Admit(context.Background(), binarySource(planpb.BinaryExpr_LogicalAnd, trueSource(), source))
			require.Nil(t, admitted)
			assertFailure(t, err, tc.kind, translation.StageAdmission)
		})
	}
	for _, tc := range []struct {
		typ      schemapb.DataType
		nullable bool
		op       planpb.OpType
		value    *planpb.GenericValue
	}{
		{schemapb.DataType_Int8, false, planpb.OpType_GreaterThan, integerSource(300)},
		{schemapb.DataType_Double, false, planpb.OpType_Equal, &planpb.GenericValue{Val: &planpb.GenericValue_FloatVal{FloatVal: 1}}},
		{schemapb.DataType_Int64, true, planpb.OpType_Equal, integerSource(0)},
		{schemapb.DataType_Bool, false, planpb.OpType_LessThan, &planpb.GenericValue{Val: &planpb.GenericValue_BoolVal{}}},
	} {
		translator, null, _ := opaqueFixture(t, tc.typ, tc.nullable)
		admitted, err := translator.Admit(context.Background(), unarySource(null.GetNullExpr().ColumnInfo, tc.op, tc.value))
		require.Nil(t, admitted)
		assertFailure(t, err, translation.UnsupportedSource, translation.StageAdmission)
	}
}

func TestUnaryComparisonGraphAndTranslationDefects(t *testing.T) {
	ctx := context.Background()
	translator, null, _ := opaqueFixture(t, schemapb.DataType_Int64, false)
	source := unarySource(null.GetNullExpr().ColumnInfo, planpb.OpType_GreaterThan, integerSource(7))
	admitted, imported := importSource(t, translator, source)
	call := imported.Root().(*expressions.CallExpression)
	runtime, err := adapterruntime.SharedRuntime()
	require.NoError(t, err)
	changedValue, err := runtime.ValueFactory().NewInteger(runtime.Types().Int64NonNull, 8)
	require.NoError(t, err)
	literal, err := expressions.NewLiteralExpression(changedValue)
	require.NoError(t, err)
	changedHandle, err := runtime.ComparisonFunction(planpb.OpType_LessThan, call.ChildAt(0).Type(), call.ChildAt(1).Type())
	require.NoError(t, err)
	changedOp, err := expressions.NewCallExpression(call.Context(), changedHandle, call.Children())
	require.NoError(t, err)
	foreignContext, err := expressions.NewCallExpression(&noEvaluationContext{runtime.FunctionContext()}, call.Handle(), call.Children())
	require.NoError(t, err)
	otherImport, err := translator.Import(ctx, admitted)
	require.NoError(t, err)
	foreignReference := otherImport.Root().ChildAt(0)
	for _, graph := range []expressions.Expression{
		changedOp, foreignContext,
		rebuiltCall(t, call, call.ChildAt(1), call.ChildAt(0)),
		rebuiltCall(t, call, call.ChildAt(0), literal),
		rebuiltCall(t, call, foreignReference, call.ChildAt(1)),
	} {
		output, err := translator.Export(ctx, imported, graph)
		require.Nil(t, output)
		assertFailure(t, err, translation.TranslationDefect, translation.StageExport)
	}
	for _, mutate := range []func(*planpb.Expr){
		func(e *planpb.Expr) { e.GetUnaryRangeExpr().Op = planpb.OpType_LessThan },
		func(e *planpb.Expr) { e.GetUnaryRangeExpr().TemplateVariableName = "changed" },
		func(e *planpb.Expr) { e.IsTemplate = !e.IsTemplate },
		func(e *planpb.Expr) { e.GetUnaryRangeExpr().Value = integerSource(8) },
		func(e *planpb.Expr) { e.GetUnaryRangeExpr().ColumnInfo.FieldId++ },
		func(e *planpb.Expr) { e.GetUnaryRangeExpr().ExtraValues = []*planpb.GenericValue{integerSource(1)} },
		func(e *planpb.Expr) { e.GetUnaryRangeExpr().ProtoReflect().SetUnknown([]byte{0xa0, 0x06, 1}) },
	} {
		output, err := translator.Export(ctx, imported, call)
		require.NoError(t, err)
		mutate(output)
		assertFailure(t, translator.ValidateCandidate(ctx, imported, output), translation.TranslationDefect, translation.StageCandidateValidation)
	}
}

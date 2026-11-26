package translation_test

import (
	"context"
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

func TestGraphExportChangedOperandsAndNaryLogic(t *testing.T) {
	translator, source := compareFixture(t, schemapb.DataType_Int64, schemapb.DataType_Int64, false)
	before := proto.Clone(source)
	imported := importSource(t, translator, source)
	runtime, err := adapterruntime.SharedRuntime()
	require.NoError(t, err)
	ref := imported.Root().ChildAt(0)
	makeInteger := func(n int64) expressions.Expression {
		value, err := runtime.ValueFactory().NewInteger(runtime.Types().Int64NonNull, n)
		require.NoError(t, err)
		literal, err := expressions.NewLiteralExpression(value)
		require.NoError(t, err)
		return literal
	}
	// All six operators are recovered from the new handle, including reversed inequalities.
	for _, tc := range []struct{ op, reverse planpb.OpType }{
		{planpb.OpType_Equal, planpb.OpType_Equal}, {planpb.OpType_NotEqual, planpb.OpType_NotEqual},
		{planpb.OpType_LessThan, planpb.OpType_GreaterThan}, {planpb.OpType_LessEqual, planpb.OpType_GreaterEqual},
		{planpb.OpType_GreaterThan, planpb.OpType_LessThan}, {planpb.OpType_GreaterEqual, planpb.OpType_LessEqual},
	} {
		handle, err := runtime.ComparisonFunction(tc.op, ref.Type(), ref.Type())
		require.NoError(t, err)
		call, err := expressions.NewCallExpression(runtime.FunctionContext(), handle, []expressions.Expression{makeInteger(9007199254740993), ref})
		require.NoError(t, err)
		output, err := translator.Export(context.Background(), imported, call)
		require.NoError(t, err)
		expected := unarySource(source.GetCompareExpr().LeftColumnInfo, tc.reverse, integerSource(9007199254740993))
		require.True(t, proto.Equal(expected, output))
	}
	args := []expressions.Expression{ref, makeInteger(9), makeInteger(-1), makeInteger(9), makeInteger(3)}
	in, err := expressions.NewSpecialFormExpression(foundation.SpecialFormIn, runtime.Types().BoolNonNull, args)
	require.NoError(t, err)
	value, err := runtime.ValueFactory().NewBool(runtime.Types().BoolNonNull, false)
	require.NoError(t, err)
	falseGraph, err := expressions.NewLiteralExpression(value)
	require.NoError(t, err)
	// Five operands exercise uneven balanced lowering and repeated graph nodes.
	graph := logicalGraph(t, foundation.Or, falseGraph, in, falseGraph, in, falseGraph)
	output, err := translator.Export(context.Background(), imported, graph)
	require.NoError(t, err)
	var leaves []*planpb.Expr
	var flatten func(*planpb.Expr) int
	flatten = func(e *planpb.Expr) int {
		if b := e.GetBinaryExpr(); b != nil {
			require.Equal(t, planpb.BinaryExpr_LogicalOr, b.Op)
			left, right := flatten(b.Left), flatten(b.Right)
			return 1 + max(left, right)
		}
		leaves = append(leaves, e)
		return 0
	}
	require.Equal(t, 3, flatten(output))
	require.Len(t, leaves, 5)
	for _, i := range []int{0, 2, 4} {
		require.True(t, proto.Equal(notSource(trueSource()), leaves[i]))
	}
	for _, i := range []int{1, 3} {
		values := leaves[i].GetTermExpr().Values
		require.Equal(t, []int64{-1, 3, 9, 9}, []int64{values[0].GetInt64Val(), values[1].GetInt64Val(), values[2].GetInt64Val(), values[3].GetInt64Val()})
	}
	for _, tc := range []struct {
		n    int64
		want bool
	}{{-1, true}, {3, true}, {9, true}, {0, false}, {10, false}} {
		require.Equal(t, tc.want, evaluatePredicate(t, output, map[int64]*planpb.GenericValue{101: integerSource(tc.n)}))
	}
	require.True(t, in.ChildAt(1).EqualsTo(args[1]), "sorting must not mutate graph order")
	leaves[1].GetTermExpr().Values[0].Val = &planpb.GenericValue_Int64Val{Int64Val: 999}
	require.Equal(t, int64(-1), leaves[3].GetTermExpr().Values[0].GetInt64Val())
	require.True(t, proto.Equal(before, source))
}

func TestGraphExportCoverageIsAtomic(t *testing.T) {
	translator, source := compareFixture(t, schemapb.DataType_Int64, schemapb.DataType_Int64, false)
	imported := importSource(t, translator, source)
	runtime, err := adapterruntime.SharedRuntime()
	require.NoError(t, err)
	value, err := runtime.ValueFactory().NewInteger(runtime.Types().Int64NonNull, 7)
	require.NoError(t, err)
	literal, err := expressions.NewLiteralExpression(value)
	require.NoError(t, err)
	handle, err := runtime.ComparisonFunction(planpb.OpType_Equal, literal.Type(), literal.Type())
	require.NoError(t, err)
	unsupported, err := expressions.NewCallExpression(runtime.FunctionContext(), handle, []expressions.Expression{literal, literal})
	require.NoError(t, err)
	// A valid call unsupported by native export occurs after an exportable sibling.
	graph := logicalGraph(t, foundation.And, imported.Root(), unsupported)
	output, err := translator.Export(context.Background(), imported, graph)
	require.Nil(t, output)
	assertFailure(t, err, translation.UnsupportedExpression, translation.StageExport)
	// The failed attempt leaves all bindings usable.
	output, err = translator.Export(context.Background(), imported, imported.Root())
	require.NoError(t, err)
	require.True(t, proto.Equal(expectedExport(source), output))
}

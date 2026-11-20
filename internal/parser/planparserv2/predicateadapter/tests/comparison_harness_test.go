package predicateadapter_test

import (
	"math"
	"testing"

	"github.com/lie-yan/predicate-optimization/expressions"
	"github.com/lie-yan/predicate-optimization/foundation"
	shared "github.com/lie-yan/predicate-optimization/functions"
	"github.com/lie-yan/predicate-optimization/types"
	valueinterfaces "github.com/lie-yan/predicate-optimization/values/interfaces"
	"github.com/lie-yan/predicate-optimization/values/interfaces/tests/testutil"
	"github.com/stretchr/testify/require"

	"github.com/milvus-io/milvus/internal/parser/planparserv2/predicateadapter/functions"
	"github.com/milvus-io/milvus/internal/parser/planparserv2/predicateadapter/values"
)

// Isolated executable fixture; no registration or callback is published to production.
type comparisonHarness struct {
	catalog *functions.FunctionCatalog
	context *functions.FunctionContext
	integer *types.IntType
	boolean *types.BoolType
	calls   int
}

func newComparisonHarness(t *testing.T) *comparisonHarness {
	t.Helper()
	integer, err := types.NewIntType(types.Int64, foundation.NonNull)
	require.NoError(t, err)
	boolean, err := types.NewBoolType(foundation.NonNull)
	require.NoError(t, err)
	definitions, err := functions.ComparisonDefinitions(integer, boolean)
	require.NoError(t, err)
	catalogBuilder := functions.NewFunctionCatalogBuilder()
	for _, def := range definitions {
		require.NoError(t, catalogBuilder.Register(def))
	}
	catalog, err := catalogBuilder.Build()
	require.NoError(t, err)
	h := &comparisonHarness{catalog: catalog, integer: integer, boolean: boolean}
	builder := functions.NewFunctionContextBuilder(catalog)
	for _, def := range definitions {
		handle, err := catalog.LookupFunction(def.Name, def.Metadata.Signature().ArgumentTypes())
		require.NoError(t, err)
		op, _ := def.Metadata.Operator().Get()
		isInteger := def.Metadata.Signature().ArgumentTypeAt(0).EqualsTo(integer)
		require.NoError(t, builder.BindImplementation(handle, func(args []valueinterfaces.Value) (valueinterfaces.Value, error) {
			h.calls++
			var result bool
			if isInteger {
				left, right := args[0].(valueinterfaces.IntegerValue).Integer(), args[1].(valueinterfaces.IntegerValue).Integer()
				switch op {
				case foundation.Equal:
					result = left == right
				case foundation.NotEqual:
					result = left != right
				case foundation.LessThan:
					result = left < right
				case foundation.LessThanOrEqual:
					result = left <= right
				case foundation.GreaterThan:
					result = left > right
				case foundation.GreaterThanOrEqual:
					result = left >= right
				default:
					t.Fatalf("unexpected integer comparison: %v", op)
				}
			} else {
				left, right := args[0].(valueinterfaces.BoolValue).Bool(), args[1].(valueinterfaces.BoolValue).Bool()
				switch op {
				case foundation.Equal:
					result = left == right
				case foundation.NotEqual:
					result = left != right
				default:
					t.Fatalf("unexpected Boolean comparison: %v", op)
				}
			}
			return values.Factory.NewBool(boolean, result)
		}))
	}
	h.context, err = builder.Build()
	require.NoError(t, err)
	return h
}

func (h *comparisonHarness) evaluate(t *testing.T, op foundation.OperatorType, left, right valueinterfaces.Value) valueinterfaces.Value {
	t.Helper()
	handle, err := h.context.ComparisonFunction(op, left.Type(), right.Type())
	require.NoError(t, err)
	metadata, err := h.context.Metadata(handle)
	require.NoError(t, err)
	signature, err := shared.BindFunctionSignature(metadata, []types.Type{left.Type(), right.Type()})
	require.NoError(t, err)
	result, err := h.context.Evaluate(handle, signature, []valueinterfaces.Value{left, right})
	require.NoError(t, err)
	require.True(t, result.Type().EqualsTo(signature.ResultType()))
	return result
}

func TestHarnessIntegerComparisons(t *testing.T) {
	h := newComparisonHarness(t)
	ops := []foundation.OperatorType{foundation.Equal, foundation.NotEqual, foundation.LessThan, foundation.LessThanOrEqual, foundation.GreaterThan, foundation.GreaterThanOrEqual}
	for _, tc := range []struct {
		left, right int64
		want        [6]bool
	}{
		{math.MinInt64, math.MaxInt64, [6]bool{false, true, true, true, false, false}},
		{math.MaxInt64, math.MinInt64, [6]bool{false, true, false, false, true, true}},
		{math.MinInt64, math.MinInt64, [6]bool{true, false, false, true, false, true}},
		{math.MaxInt64, math.MaxInt64, [6]bool{true, false, false, true, false, true}},
		{-1, 0, [6]bool{false, true, true, true, false, false}},
		{1, 0, [6]bool{false, true, false, false, true, true}},
		{0, 0, [6]bool{true, false, false, true, false, true}},
		{9007199254740993, 9007199254740992, [6]bool{false, true, false, false, true, true}},
	} {
		left, err := values.Factory.NewInteger(h.integer, tc.left)
		require.NoError(t, err)
		right, err := values.Factory.NewInteger(h.integer, tc.right)
		require.NoError(t, err)
		for i, op := range ops {
			for _, arg := range []valueinterfaces.Value{left, testutil.WrapScalar(left)} {
				result := h.evaluate(t, op, arg, right)
				require.Equal(t, tc.want[i], result.(valueinterfaces.BoolValue).Bool(), "%d %v %d", tc.left, op, tc.right)
			}
		}
	}
}

func TestHarnessBooleanComparisons(t *testing.T) {
	h := newComparisonHarness(t)
	for _, tc := range []struct{ left, right, equal bool }{
		{false, false, true}, {false, true, false}, {true, false, false}, {true, true, true},
	} {
		left, err := values.Factory.NewBool(h.boolean, tc.left)
		require.NoError(t, err)
		right, err := values.Factory.NewBool(h.boolean, tc.right)
		require.NoError(t, err)
		for _, arg := range []valueinterfaces.Value{left, testutil.WrapScalar(left)} {
			require.Equal(t, tc.equal, h.evaluate(t, foundation.Equal, arg, right).(valueinterfaces.BoolValue).Bool())
			require.Equal(t, !tc.equal, h.evaluate(t, foundation.NotEqual, arg, right).(valueinterfaces.BoolValue).Bool())
		}
	}
}

func TestHarnessNullableComparisonsAndIsolation(t *testing.T) {
	h := newComparisonHarness(t)
	nullable, err := types.NewIntType(types.Int64, foundation.MayNull)
	require.NoError(t, err)
	left, err := values.Factory.NewInteger(nullable, 42)
	require.NoError(t, err)
	right, err := values.Factory.NewInteger(h.integer, 42)
	require.NoError(t, err)
	require.True(t, h.evaluate(t, foundation.Equal, testutil.WrapScalar(left), right).(valueinterfaces.BoolValue).Bool())
	null, err := values.Factory.NewNull(nullable)
	require.NoError(t, err)
	before := h.calls
	for _, args := range [][2]valueinterfaces.Value{{null, right}, {right, null}, {null, null}} {
		require.True(t, h.evaluate(t, foundation.Equal, args[0], args[1]).IsNull())
	}
	require.Equal(t, before, h.calls, "NULL propagation must skip real implementations")
	// A separate catalog models production ownership without publishing a runtime.
	definitions, err := functions.ComparisonDefinitions(h.integer, h.boolean)
	require.NoError(t, err)
	builder := functions.NewFunctionCatalogBuilder()
	for _, def := range definitions {
		require.NoError(t, builder.Register(def))
	}
	catalog, err := builder.Build()
	require.NoError(t, err)
	production, err := functions.NewFunctionContextBuilder(catalog).Build()
	require.NoError(t, err)
	handle, err := h.context.ComparisonFunction(foundation.Equal, h.integer, h.integer)
	require.NoError(t, err)
	literal, err := expressions.NewLiteralExpression(right)
	require.NoError(t, err)
	_, err = expressions.NewCallExpression(production, handle, []expressions.Expression{literal, literal})
	require.ErrorIs(t, err, shared.ErrIncompatibleHandle)
	productionHandle, err := production.ComparisonFunction(foundation.Equal, h.integer, h.integer)
	require.NoError(t, err)
	_, err = expressions.NewCallExpression(h.context, productionHandle, []expressions.Expression{literal, literal})
	require.ErrorIs(t, err, shared.ErrIncompatibleHandle)
}

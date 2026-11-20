package functions

import (
	"testing"

	"github.com/lie-yan/predicate-optimization/foundation"
	shared "github.com/lie-yan/predicate-optimization/functions"
	"github.com/lie-yan/predicate-optimization/types"
	"github.com/stretchr/testify/require"
)

func TestInitialComparisonDefinitions(t *testing.T) {
	integer, err := types.NewIntType(types.Int64, foundation.NonNull)
	require.NoError(t, err)
	boolean, err := types.NewBoolType(foundation.NonNull)
	require.NoError(t, err)
	definitions, err := ComparisonDefinitions(integer, boolean)
	require.NoError(t, err)
	require.Len(t, definitions, 8)
	catalog := catalogWith(t, definitions...)
	for _, tc := range []struct {
		name    string
		op      foundation.OperatorType
		operand types.Type
	}{
		{"i64.eq", foundation.Equal, integer}, {"i64.ne", foundation.NotEqual, integer},
		{"i64.lt", foundation.LessThan, integer}, {"i64.le", foundation.LessThanOrEqual, integer},
		{"i64.gt", foundation.GreaterThan, integer}, {"i64.ge", foundation.GreaterThanOrEqual, integer},
		{"bool.eq", foundation.Equal, boolean}, {"bool.ne", foundation.NotEqual, boolean},
	} {
		t.Run(tc.name, func(t *testing.T) {
			handle, err := catalog.ComparisonFunction(tc.op, tc.operand, tc.operand)
			require.NoError(t, err)
			named, err := catalog.LookupFunction(FunctionName{Namespace: "milvus.scalar.v1", Name: tc.name}, []types.Type{tc.operand, tc.operand})
			require.NoError(t, err)
			require.True(t, handle.EqualsTo(named))
			metadata, err := catalog.Metadata(handle)
			require.NoError(t, err)
			require.Same(t, tc.operand, metadata.Signature().ArgumentTypeAt(0))
			require.Same(t, tc.operand, metadata.Signature().ArgumentTypeAt(1))
			require.Same(t, boolean, metadata.Signature().ResultType())
			require.Equal(t, []shared.NullPolicy{shared.PropagateNull, shared.PropagateNull}, metadata.NullPolicies())
			_, present := metadata.BuiltInName().Get()
			require.False(t, present)
		})
	}
	handle, err := catalog.ComparisonFunction(foundation.LessThan, boolean, boolean)
	require.ErrorIs(t, err, shared.ErrUnsupportedLookup)
	require.Nil(t, handle)
}

func TestInitialComparisonsInMixedCatalog(t *testing.T) {
	integer, err := types.NewIntType(types.Int64, foundation.NonNull)
	require.NoError(t, err)
	boolean, err := types.NewBoolType(foundation.NonNull)
	require.NoError(t, err)
	comparisons, err := ComparisonDefinitions(integer, boolean)
	require.NoError(t, err)
	add := registration(t, "add", foundation.Add, "", shared.PropagateNull)
	ordinary := registration(t, "ordinary", 0, "ordinary_builtin", shared.PropagateNull)
	builder := NewFunctionCatalogBuilder()
	require.NoError(t, builder.Register(add))
	for _, def := range comparisons {
		require.NoError(t, builder.Register(def))
	}
	require.NoError(t, builder.Register(ordinary))
	// Duplicate declarations and registration order must not alter alias identity.
	require.NoError(t, builder.Register(comparisons[0]))
	catalog, err := builder.Build()
	require.NoError(t, err)
	args := []types.Type{integer, integer}
	arithmetic, err := catalog.ArithmeticFunction(foundation.Add, args)
	require.NoError(t, err)
	builtin, err := catalog.LookupBuiltInFunction("ordinary_builtin", args)
	require.NoError(t, err)
	named, err := catalog.LookupFunction(ordinary.Name, args)
	require.NoError(t, err)
	require.True(t, builtin.EqualsTo(named))
	comparison, err := catalog.ComparisonFunction(foundation.Equal, integer, integer)
	require.NoError(t, err)
	require.False(t, comparison.EqualsTo(arithmetic))
	require.False(t, comparison.EqualsTo(builtin))
	nullable, err := integer.WithNullability(foundation.MayNull)
	require.NoError(t, err)
	adapted, err := catalog.ComparisonFunction(foundation.Equal, nullable, integer)
	require.NoError(t, err)
	require.True(t, comparison.EqualsTo(adapted))
	metadata, err := catalog.Metadata(adapted)
	require.NoError(t, err)
	call, err := shared.BindFunctionSignature(metadata, []types.Type{nullable, integer})
	require.NoError(t, err)
	require.Equal(t, foundation.MayNull, call.ResultType().Nullability())
	require.Same(t, boolean, metadata.Signature().ResultType())
}

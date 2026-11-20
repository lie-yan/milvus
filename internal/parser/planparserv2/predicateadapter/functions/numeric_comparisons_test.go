package functions

import (
	"testing"

	"github.com/lie-yan/predicate-optimization/foundation"
	shared "github.com/lie-yan/predicate-optimization/functions"
	"github.com/lie-yan/predicate-optimization/types"
	"github.com/stretchr/testify/require"
)

func TestNumericComparisonBinding(t *testing.T) {
	boolean, err := types.NewBoolType(foundation.NonNull)
	require.NoError(t, err)
	narrow, err := types.NewIntType(types.Int8, foundation.NonNull)
	require.NoError(t, err)
	wide, err := types.NewFloatType(types.Float64, foundation.NonNull)
	require.NoError(t, err)
	definitions, err := NumericComparisonDefinitions(boolean, narrow, wide)
	require.NoError(t, err)
	catalog := catalogWith(t, definitions...)
	// Distinct same-Type overloads must not enable implicit mixed-domain comparisons.
	_, err = catalog.ComparisonFunction(foundation.Equal, narrow, wide)
	require.ErrorIs(t, err, shared.ErrUnsupportedLookup)
	for _, operand := range []types.Type{narrow, wide} {
		nullable, err := operand.WithNullability(foundation.MayNull)
		require.NoError(t, err)
		handle, err := catalog.ComparisonFunction(foundation.LessThan, nullable, operand)
		require.NoError(t, err)
		metadata, err := catalog.Metadata(handle)
		require.NoError(t, err)
		signature, err := shared.BindFunctionSignature(metadata, []types.Type{nullable, operand})
		require.NoError(t, err)
		require.Equal(t, foundation.MayNull, signature.ResultType().Nullability())
		require.True(t, signature.ArgumentTypeAt(0).EqualsTo(nullable))
		require.True(t, signature.ArgumentTypeAt(1).EqualsTo(operand))
	}
}

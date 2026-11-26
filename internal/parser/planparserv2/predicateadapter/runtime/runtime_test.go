package runtime

import (
	"testing"

	"github.com/lie-yan/predicate-optimization/foundation"
	shared "github.com/lie-yan/predicate-optimization/functions"
	"github.com/lie-yan/predicate-optimization/types"
	"github.com/stretchr/testify/require"

	"github.com/milvus-io/milvus/internal/parser/planparserv2/predicateadapter/functions"
	"github.com/milvus-io/milvus/pkg/v3/proto/planpb"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
)

func TestRuntimeConstructionAndMapping(t *testing.T) {
	runtime, err := NewRuntime()
	require.NoError(t, err)
	primitives := runtime.Types()
	for _, tc := range []struct {
		milvus    planpb.OpType
		optimizer foundation.OperatorType
	}{
		{planpb.OpType_Equal, foundation.Equal}, {planpb.OpType_NotEqual, foundation.NotEqual},
		{planpb.OpType_LessThan, foundation.LessThan}, {planpb.OpType_LessEqual, foundation.LessThanOrEqual},
		{planpb.OpType_GreaterThan, foundation.GreaterThan}, {planpb.OpType_GreaterEqual, foundation.GreaterThanOrEqual},
	} {
		operands := []types.Type{primitives.Int64NonNull}
		if tc.milvus == planpb.OpType_Equal || tc.milvus == planpb.OpType_NotEqual {
			operands = append(operands, primitives.BoolNonNull)
		}
		for _, operand := range operands {
			handle, err := runtime.ComparisonFunction(tc.milvus, operand, operand)
			require.NoError(t, err)
			providerHandle, err := runtime.FunctionContext().ComparisonFunction(tc.optimizer, operand, operand)
			require.NoError(t, err)
			require.True(t, handle.EqualsTo(providerHandle))
			metadata, err := runtime.FunctionContext().Metadata(handle)
			require.NoError(t, err)
			catalogMetadata, err := runtime.Catalog().Metadata(handle)
			require.NoError(t, err)
			require.Same(t, catalogMetadata, metadata)
			require.Same(t, operand, metadata.Signature().ArgumentTypeAt(0))
			require.Same(t, primitives.BoolNonNull, metadata.Signature().ResultType())
			mapped, err := runtime.ComparisonOperator(handle)
			require.NoError(t, err)
			require.Equal(t, tc.milvus, mapped)
		}
	}
	value, err := runtime.ValueFactory().NewInteger(primitives.Int64NonNull, 42)
	require.NoError(t, err)
	require.Same(t, primitives.Int64NonNull, value.Type())
	primitives.Int64NonNull = nil
	require.Same(t, int64NonNullType, runtime.Types().Int64NonNull)
	foreign, err := NewRuntime()
	require.NoError(t, err)
	handle, err := foreign.ComparisonFunction(planpb.OpType_Equal, int64NonNullType, int64NonNullType)
	require.NoError(t, err)
	_, err = runtime.ComparisonOperator(handle)
	require.ErrorIs(t, err, shared.ErrIncompatibleHandle)
	_, err = runtime.ComparisonFunction(planpb.OpType_Invalid, int64NonNullType, int64NonNullType)
	require.ErrorIs(t, err, shared.ErrUnsupportedLookup)
}

func TestRuntimeSetupFailureDoesNotEscape(t *testing.T) {
	definitions, err := functions.ComparisonDefinitions(int64NonNullType, boolNonNullType)
	require.NoError(t, err)
	invalid := append([]functions.FunctionDefinition(nil), definitions...)
	invalid[len(invalid)-1].Metadata = nil
	// Registration must fail without returning a partial runtime.
	runtime, err := assembleRuntime(invalid)
	require.ErrorIs(t, err, merr.ErrServiceInternal)
	require.Nil(t, runtime)
	// Different names defining the same operator/signature are ambiguous in the future importer.
	ambiguous := append([]functions.FunctionDefinition(nil), definitions...)
	duplicate := definitions[0]
	duplicate.Name.Name = "another.eq"
	ambiguous = append(ambiguous, duplicate)
	runtime, err = assembleRuntime(ambiguous)
	require.ErrorIs(t, err, shared.ErrUnsupportedLookup)
	require.Nil(t, runtime)
	// The initial Bool/Int64 definitions alone cannot satisfy wider translation.
	runtime, err = assembleRuntime(definitions)
	require.ErrorIs(t, err, shared.ErrUnsupportedLookup)
	require.Nil(t, runtime)
	// A failed attempt does not poison later construction.
	runtime, err = NewRuntime()
	require.NoError(t, err)
	require.NotNil(t, runtime)
}

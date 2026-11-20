package functions

import (
	"testing"

	"github.com/lie-yan/predicate-optimization/expressions"
	"github.com/lie-yan/predicate-optimization/foundation"
	shared "github.com/lie-yan/predicate-optimization/functions"
	functioninterfaces "github.com/lie-yan/predicate-optimization/functions/interfaces"
	"github.com/lie-yan/predicate-optimization/types"
	valueinterfaces "github.com/lie-yan/predicate-optimization/values/interfaces"
	"github.com/stretchr/testify/require"

	"github.com/milvus-io/milvus/internal/parser/planparserv2/predicateadapter/values"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
)

func TestContextCatalogDelegation(t *testing.T) {
	less := registration(t, "less", foundation.LessThan, "", shared.PropagateNull)
	add := registration(t, "add", foundation.Add, "", shared.PropagateNull)
	ordinary := registration(t, "ordinary", 0, "ordinary", shared.PropagateNull)
	catalog := catalogWith(t, less, add, ordinary)
	context, err := NewFunctionContextBuilder(catalog).Build()
	require.NoError(t, err)
	var provider functioninterfaces.FunctionContext = context
	args := less.Metadata.Signature().ArgumentTypes()
	for _, tc := range []struct {
		definition             FunctionDefinition
		lookup                 func() (functioninterfaces.FunctionHandle, error)
		comparison, arithmetic bool
	}{
		{less, func() (functioninterfaces.FunctionHandle, error) {
			return provider.ComparisonFunction(foundation.LessThan, args[0], args[1])
		}, true, false},
		{add, func() (functioninterfaces.FunctionHandle, error) {
			return provider.ArithmeticFunction(foundation.Add, args)
		}, false, true},
		{ordinary, func() (functioninterfaces.FunctionHandle, error) {
			return provider.LookupBuiltInFunction("ordinary", args)
		}, false, false},
	} {
		t.Run(tc.definition.Name.Name, func(t *testing.T) {
			handle, err := tc.lookup()
			require.NoError(t, err)
			named, err := catalog.LookupFunction(tc.definition.Name, args)
			require.NoError(t, err)
			require.True(t, handle.EqualsTo(named))
			metadata, err := provider.Metadata(handle)
			require.NoError(t, err)
			require.Same(t, tc.definition.Metadata, metadata)
			comparison, err := provider.IsComparisonFunction(handle)
			require.NoError(t, err)
			require.Equal(t, tc.comparison, comparison)
			arithmetic, err := provider.IsArithmeticFunction(handle)
			require.NoError(t, err)
			require.Equal(t, tc.arithmetic, arithmetic)
		})
	}
	_, err = provider.LookupBuiltInFunction("missing", args)
	require.ErrorIs(t, err, shared.ErrUnsupportedLookup)
	foreign := catalogWith(t, less).handleAt(0)
	_, err = provider.Metadata(foreign)
	require.ErrorIs(t, err, shared.ErrIncompatibleHandle)
	for _, classify := range []func(functioninterfaces.FunctionHandle) (bool, error){provider.IsComparisonFunction, provider.IsArithmeticFunction} {
		_, err = classify(foreign)
		require.ErrorIs(t, err, shared.ErrIncompatibleHandle)
	}
}

func TestContextBindingsAndCallOwnership(t *testing.T) {
	boolean, _ := types.NewBoolType(foundation.NonNull)
	catalog := evaluationCatalog(t, shared.PropagateNull, boolean, boolean)
	handle := catalog.handleAt(0)
	input, err := values.Factory.NewBool(boolean, false)
	require.NoError(t, err)
	literal, err := expressions.NewLiteralExpression(input)
	require.NoError(t, err)
	var calls []*expressions.CallExpression
	for _, truth := range []bool{false, true} {
		builder := NewFunctionContextBuilder(catalog)
		copyOfBuilder := *builder
		implementation := func([]valueinterfaces.Value) (valueinterfaces.Value, error) {
			return values.Factory.NewBool(boolean, truth)
		}
		require.NoError(t, builder.BindImplementation(handle, implementation))
		context, err := copyOfBuilder.Build()
		require.NoError(t, err)
		require.ErrorIs(t, builder.BindImplementation(handle, implementation), merr.ErrServiceInternal)
		_, err = builder.Build()
		require.ErrorIs(t, err, merr.ErrServiceInternal)
		call, err := expressions.NewCallExpression(context, handle, []expressions.Expression{literal})
		require.NoError(t, err)
		require.Same(t, context, call.Context())
		calls = append(calls, call)
	}
	// Building another context sharing the catalog cannot change a retained call's execution policy.
	for i, call := range calls {
		result, err := call.Context().Evaluate(call.Handle(), call.Signature(), []valueinterfaces.Value{input})
		require.NoError(t, err)
		require.Equal(t, i == 1, result.(valueinterfaces.BoolValue).Bool())
	}
	unbound, err := NewFunctionContextBuilder(catalog).Build()
	require.NoError(t, err)
	_, err = unbound.Evaluate(handle, calls[0].Signature(), []valueinterfaces.Value{input})
	require.ErrorIs(t, err, shared.ErrEvaluationUnavailable)
	foreignCatalog := evaluationCatalog(t, shared.PropagateNull, boolean, boolean)
	foreign, err := NewFunctionContextBuilder(foreignCatalog).Build()
	require.NoError(t, err)
	_, err = foreign.Evaluate(handle, calls[0].Signature(), []valueinterfaces.Value{input})
	require.ErrorIs(t, err, shared.ErrIncompatibleHandle)
	_, err = calls[0].Context().Metadata(foreignCatalog.handleAt(0))
	require.ErrorIs(t, err, shared.ErrIncompatibleHandle)
}

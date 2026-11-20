package functions

import (
	"testing"

	"github.com/lie-yan/predicate-optimization/foundation"
	shared "github.com/lie-yan/predicate-optimization/functions"
	functioninterfaces "github.com/lie-yan/predicate-optimization/functions/interfaces"
	"github.com/lie-yan/predicate-optimization/types"
	"github.com/stretchr/testify/require"
)

func TestFunctionRoutesAndClassification(t *testing.T) {
	less := registration(t, "less", foundation.LessThan, "builtin_less", shared.PropagateNull)
	add := registration(t, "add", foundation.Add, "builtin_add", shared.PropagateNull)
	ordinary := registration(t, "ordinary", 0, "builtin_other", shared.PropagateNull)
	// A suggestive name alone must not establish operator or built-in classification.
	named := registration(t, "Negate", 0, "", shared.PropagateNull)
	catalog := catalogWith(t, less, add, ordinary, named)
	for _, tc := range []struct {
		def                    FunctionDefinition
		comparison, arithmetic bool
	}{{less, true, false}, {add, false, true}, {ordinary, false, false}, {named, false, false}} {
		t.Run(tc.def.Name.Name, func(t *testing.T) {
			args := tc.def.Metadata.Signature().ArgumentTypes()
			namedHandle, err := catalog.LookupFunction(tc.def.Name, args)
			require.NoError(t, err)
			if builtin, present := tc.def.Metadata.BuiltInName().Get(); present {
				handle, err := catalog.LookupBuiltInFunction(builtin, args)
				require.NoError(t, err)
				require.True(t, namedHandle.EqualsTo(handle))
			}
			if op, present := tc.def.Metadata.Operator().Get(); present {
				var handle functioninterfaces.FunctionHandle
				if tc.comparison {
					handle, err = catalog.ComparisonFunction(op, args[0], args[1])
				} else {
					handle, err = catalog.ArithmeticFunction(op, args)
				}
				require.NoError(t, err)
				require.True(t, namedHandle.EqualsTo(handle))
				metadata, err := catalog.Metadata(handle)
				require.NoError(t, err)
				require.Equal(t, tc.def.Metadata.Operator(), metadata.Operator())
				require.Equal(t, tc.def.Metadata.BuiltInName(), metadata.BuiltInName())
			}
			comparison, err := catalog.IsComparisonFunction(namedHandle)
			require.NoError(t, err)
			require.Equal(t, tc.comparison, comparison)
			arithmetic, err := catalog.IsArithmeticFunction(namedHandle)
			require.NoError(t, err)
			require.Equal(t, tc.arithmetic, arithmetic)
		})
	}
}

func TestOperatorLookupValidation(t *testing.T) {
	add := registration(t, "add", foundation.Add, "", shared.PropagateNull)
	integer := add.Metadata.Signature().ArgumentTypeAt(0)
	signature, err := shared.NewFunctionSignature([]types.Type{integer}, integer)
	require.NoError(t, err)
	metadata, err := shared.NewFunctionMetadata(signature, "negate", foundation.Optional[shared.BuiltInFunctionName]{}, foundation.Some(foundation.Negate), []shared.NullPolicy{shared.PropagateNull})
	require.NoError(t, err)
	negate := FunctionDefinition{Name: FunctionName{Namespace: "test.scalar", Name: "negate"}, Metadata: metadata}
	catalog := catalogWith(t, add, negate)
	handle, err := catalog.ArithmeticFunction(foundation.Negate, []types.Type{integer})
	require.NoError(t, err)
	require.True(t, handle.EqualsTo(catalog.handleAt(1)))
	for name, lookup := range map[string]func() (functioninterfaces.FunctionHandle, error){
		"comparison family": func() (functioninterfaces.FunctionHandle, error) {
			return catalog.ComparisonFunction(foundation.Add, integer, integer)
		},
		"arithmetic family": func() (functioninterfaces.FunctionHandle, error) {
			return catalog.ArithmeticFunction(foundation.LessThan, []types.Type{integer, integer})
		},
		"binary arity": func() (functioninterfaces.FunctionHandle, error) {
			return catalog.ArithmeticFunction(foundation.Add, []types.Type{integer})
		},
		"unary arity": func() (functioninterfaces.FunctionHandle, error) {
			return catalog.ArithmeticFunction(foundation.Negate, []types.Type{integer, integer})
		},
		"missing operator": func() (functioninterfaces.FunctionHandle, error) {
			return catalog.ComparisonFunction(foundation.Equal, integer, integer)
		},
		"name is not builtin": func() (functioninterfaces.FunctionHandle, error) {
			return catalog.LookupBuiltInFunction("add", []types.Type{integer, integer})
		},
	} {
		t.Run(name, func(t *testing.T) {
			handle, err := lookup()
			require.ErrorIs(t, err, shared.ErrUnsupportedLookup)
			require.Nil(t, handle)
		})
	}
}

func TestFunctionRoutesAmbiguityAndHandleErrors(t *testing.T) {
	first := registration(t, "first", foundation.Add, "addition", shared.PropagateNull)
	second := first
	second.Name.Name = "second"
	catalog := catalogWith(t, first, second)
	args := first.Metadata.Signature().ArgumentTypes()
	handle, err := catalog.ArithmeticFunction(foundation.Add, args)
	require.ErrorIs(t, err, shared.ErrUnsupportedLookup)
	require.Nil(t, handle)
	handle, err = catalog.LookupBuiltInFunction("addition", args)
	require.ErrorIs(t, err, shared.ErrUnsupportedLookup)
	require.Nil(t, handle)
	foreign := catalogWith(t, first).handleAt(0)
	for _, classify := range []func(functioninterfaces.FunctionHandle) (bool, error){catalog.IsComparisonFunction, catalog.IsArithmeticFunction} {
		ok, err := classify(foreign)
		require.ErrorIs(t, err, shared.ErrIncompatibleHandle)
		require.False(t, ok)
		ok, err = classify(&functionHandle{catalog: catalog, index: len(catalog.entries)})
		require.ErrorIs(t, err, shared.ErrUnknownHandle)
		require.False(t, ok)
	}
}

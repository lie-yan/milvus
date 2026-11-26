package functions

import (
	"testing"

	"github.com/lie-yan/predicate-optimization/foundation"
	shared "github.com/lie-yan/predicate-optimization/functions"
	"github.com/lie-yan/predicate-optimization/types"
	"github.com/stretchr/testify/require"
)

// Builds ordinary concrete overloads without operator semantics in the test fixture.
func overload(t *testing.T, name FunctionName, args []types.Type, result types.Type, policy shared.NullPolicy) FunctionDefinition {
	t.Helper()
	sig, err := shared.NewFunctionSignature(args, result)
	require.NoError(t, err)
	policies := make([]shared.NullPolicy, len(args))
	for i := range policies {
		policies[i] = policy
	}
	metadata, err := shared.NewFunctionMetadata(sig, "diagnostic label", foundation.Optional[shared.BuiltInFunctionName]{}, foundation.Optional[foundation.OperatorType]{}, policies)
	require.NoError(t, err)
	return FunctionDefinition{Name: name, Metadata: metadata}
}

func TestNamedLookupBinding(t *testing.T) {
	def := registration(t, "less", foundation.LessThan, "", shared.PropagateNull)
	integer, err := types.NewIntType(types.Int64, foundation.NonNull)
	require.NoError(t, err)
	nullable, err := integer.WithNullability(foundation.MayNull)
	require.NoError(t, err)
	zero := overload(t, def.Name, nil, integer, shared.PropagateNull)
	catalog := catalogWith(t, def, zero)
	for _, args := range [][]types.Type{{integer, integer}, {nullable, integer}} {
		handle, err := catalog.LookupFunction(def.Name, args)
		require.NoError(t, err)
		require.Same(t, catalog.handleAt(0), handle)
		metadata, err := catalog.Metadata(handle)
		require.NoError(t, err)
		require.Same(t, def.Metadata, metadata)
	}
	handle, err := catalog.LookupFunction(def.Name, nil)
	require.NoError(t, err)
	require.True(t, handle.EqualsTo(catalog.handleAt(1)))
}

func TestNamedLookupUnsupported(t *testing.T) {
	def := registration(t, "less", foundation.LessThan, "", shared.PropagateNull)
	integer := def.Metadata.Signature().ArgumentTypeAt(0)
	narrow, err := types.NewIntType(types.Int8, foundation.NonNull)
	require.NoError(t, err)
	nullable, err := integer.WithNullability(foundation.MayNull)
	require.NoError(t, err)
	unspecified := registration(t, "unspecified", 0, "", shared.NullPolicyUnspecified)
	catalog := catalogWith(t, def, unspecified)
	for _, tc := range []struct {
		name     string
		function FunctionName
		args     []types.Type
	}{
		{"namespace", FunctionName{Namespace: "other", Name: def.Name.Name}, []types.Type{integer, integer}},
		{"name", FunctionName{Namespace: def.Name.Namespace, Name: "missing"}, []types.Type{integer, integer}},
		{"arity", def.Name, []types.Type{integer}},
		{"width", def.Name, []types.Type{narrow, integer}},
		{"null policy", unspecified.Name, []types.Type{nullable, integer}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			handle, err := catalog.LookupFunction(tc.function, tc.args)
			require.ErrorIs(t, err, shared.ErrUnsupportedLookup)
			require.Nil(t, handle)
		})
	}
}

func TestNamedLookupAmbiguity(t *testing.T) {
	integer, err := types.NewIntType(types.Int64, foundation.NonNull)
	require.NoError(t, err)
	boolean, err := types.NewBoolType(foundation.NonNull)
	require.NoError(t, err)
	nullable, err := integer.WithNullability(foundation.MayNull)
	require.NoError(t, err)
	name := FunctionName{Namespace: "test.scalar", Name: "f"}
	first := overload(t, name, []types.Type{integer}, boolean, shared.PropagateNull)
	for _, tc := range []struct {
		name   string
		second FunctionDefinition
	}{
		{"result only", overload(t, name, []types.Type{integer}, integer, shared.PropagateNull)},
		{"outer nullability", overload(t, name, []types.Type{nullable}, boolean, shared.PropagateNull)},
	} {
		t.Run(tc.name, func(t *testing.T) {
			for _, definitions := range [][]FunctionDefinition{{first, tc.second}, {tc.second, first}} {
				catalog := catalogWith(t, definitions...)
				handle, err := catalog.LookupFunction(name, []types.Type{integer})
				require.ErrorIs(t, err, shared.ErrUnsupportedLookup)
				require.Nil(t, handle)
			}
		})
	}
}

// Injects a result-adaptation defect through the public Type contract.
type brokenResultType struct{ types.Type }

func (brokenResultType) WithNullability(foundation.Nullability) (types.Type, error) {
	return nil, shared.ErrContractViolation
}

func TestNamedLookupPreservesBindingDefect(t *testing.T) {
	integer, err := types.NewIntType(types.Int64, foundation.NonNull)
	require.NoError(t, err)
	nullable, err := integer.WithNullability(foundation.MayNull)
	require.NoError(t, err)
	name := FunctionName{Namespace: "test.scalar", Name: "broken"}
	def := overload(t, name, []types.Type{integer}, brokenResultType{integer}, shared.PropagateNull)
	handle, err := catalogWith(t, def).LookupFunction(name, []types.Type{nullable})
	require.ErrorIs(t, err, shared.ErrContractViolation)
	require.Nil(t, handle)
}

func TestIndexedLookupStructuralTypesAndFallback(t *testing.T) {
	name := FunctionName{Namespace: "test.scalar", Name: "indexed"}
	integer, err := types.NewIntType(types.Int64, foundation.NonNull)
	require.NoError(t, err)
	nullable, err := integer.WithNullability(foundation.MayNull)
	require.NoError(t, err)
	array, err := types.NewArrayType(integer, foundation.NonNull)
	require.NoError(t, err)
	bound := uint64(32)
	varchar, err := types.NewVarCharType(&bound, foundation.NonNull)
	require.NoError(t, err)
	unbounded, err := types.NewVarCharType(nil, foundation.NonNull)
	require.NoError(t, err)
	definitions := []FunctionDefinition{
		overload(t, name, []types.Type{integer}, integer, shared.PropagateNull),
		overload(t, name, []types.Type{varchar}, integer, shared.PropagateNull),
		overload(t, name, []types.Type{unbounded}, integer, shared.PropagateNull),
	}
	scalar := catalogWith(t, definitions...)
	// Separate instances and outer nullability must select the same structural overload.
	sameVarchar, err := types.NewVarCharType(&bound, foundation.MayNull)
	require.NoError(t, err)
	for i, args := range [][]types.Type{{nullable}, {sameVarchar}, {unbounded}} {
		handle, err := scalar.LookupFunction(name, args)
		require.NoError(t, err)
		require.True(t, handle.EqualsTo(scalar.handleAt(i)))
	}
	bound++
	other, err := types.NewVarCharType(&bound, foundation.NonNull)
	require.NoError(t, err)
	_, err = scalar.LookupFunction(name, []types.Type{other})
	require.ErrorIs(t, err, shared.ErrUnsupportedLookup)
	// A mixed overload set must preserve composite and higher-arity binding.
	definitions = append(definitions, overload(t, name, []types.Type{array}, integer, shared.PropagateNull), overload(t, name, []types.Type{integer, integer, integer}, integer, shared.PropagateNull))
	mixed := catalogWith(t, definitions...)
	for i, args := range [][]types.Type{{nullable}, {sameVarchar}, {unbounded}, {array}, {integer, nullable, integer}} {
		handle, err := mixed.LookupFunction(name, args)
		require.NoError(t, err)
		require.True(t, handle.EqualsTo(mixed.handleAt(i)))
	}
	nullableElementArray, err := types.NewArrayType(nullable, foundation.NonNull)
	require.NoError(t, err)
	_, err = mixed.LookupFunction(name, []types.Type{nullableElementArray})
	require.ErrorIs(t, err, shared.ErrUnsupportedLookup)
}

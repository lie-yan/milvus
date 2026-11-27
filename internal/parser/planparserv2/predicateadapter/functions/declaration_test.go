package functions

import (
	"testing"

	"github.com/lie-yan/predicate-optimization/foundation"
	shared "github.com/lie-yan/predicate-optimization/functions"
	functioninterfaces "github.com/lie-yan/predicate-optimization/functions/interfaces"
	"github.com/lie-yan/predicate-optimization/types"
	valueinterfaces "github.com/lie-yan/predicate-optimization/values/interfaces"
	"github.com/stretchr/testify/require"

	"github.com/milvus-io/milvus/internal/parser/planparserv2/predicateadapter/values"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
)

func textDeclarations(t *testing.T) []FunctionDeclaration {
	t.Helper()
	boolean, err := types.NewBoolType(foundation.NonNull)
	require.NoError(t, err)
	definitions, err := VarCharComparisonDeclarations(boolean)
	require.NoError(t, err)
	definitions[0].BuiltInName = foundation.Some(shared.BuiltInFunctionName("text_equal"))
	return definitions
}

func textType(t *testing.T, bound *uint64, nullable foundation.Nullability) *types.VarCharType {
	t.Helper()
	typ, err := types.NewVarCharType(bound, nullable)
	require.NoError(t, err)
	return typ
}

func declarationCatalog(t *testing.T, declarations ...FunctionDeclaration) *FunctionCatalog {
	t.Helper()
	builder := NewFunctionCatalogBuilder()
	for _, declaration := range declarations {
		require.NoError(t, builder.RegisterDeclaration(declaration))
	}
	catalog, err := builder.Build()
	require.NoError(t, err)
	return catalog
}

func TestParameterizedLookupAndIdentity(t *testing.T) {
	definitions := textDeclarations(t)
	catalog := declarationCatalog(t, definitions...)
	foreign := declarationCatalog(t, definitions...)
	small, large := uint64(32), uint64(64)
	var previous functioninterfaces.FunctionHandle
	for _, bound := range []*uint64{&small, &large, nil} {
		nonnull := textType(t, bound, foundation.NonNull)
		nullable := textType(t, bound, foundation.MayNull)
		named, err := catalog.LookupFunction(definitions[0].Name, []types.Type{nullable, nonnull})
		require.NoError(t, err)
		builtin, err := catalog.LookupBuiltInFunction("text_equal", []types.Type{nonnull, nonnull})
		require.NoError(t, err)
		operator, err := catalog.ComparisonFunction(foundation.Equal, nullable, nonnull)
		require.NoError(t, err)
		require.True(t, named.EqualsTo(builtin))
		require.True(t, named.EqualsTo(operator))
		if previous != nil {
			require.False(t, named.EqualsTo(previous))
		}
		previous = named
		metadata, err := catalog.Metadata(named)
		require.NoError(t, err)
		require.Same(t, metadata.Signature(), named.Signature())
		require.True(t, metadata.Signature().ArgumentTypeAt(0).EqualsTo(nonnull))
		effective, err := shared.BindFunctionSignature(metadata, []types.Type{nullable, nonnull})
		require.NoError(t, err)
		require.Equal(t, foundation.MayNull, effective.ResultType().Nullability())
		comparison, err := catalog.IsComparisonFunction(named)
		require.NoError(t, err)
		require.True(t, comparison)
		_, err = foreign.Metadata(named)
		require.ErrorIs(t, err, shared.ErrIncompatibleHandle)
		ne, err := catalog.ComparisonFunction(foundation.NotEqual, nonnull, nonnull)
		require.NoError(t, err)
		require.False(t, ne.EqualsTo(named))
	}
	left, right := textType(t, &small, foundation.NonNull), textType(t, &large, foundation.NonNull)
	for _, resolve := range []func() (functioninterfaces.FunctionHandle, error){
		func() (functioninterfaces.FunctionHandle, error) {
			return catalog.LookupFunction(definitions[0].Name, []types.Type{left, right})
		},
		func() (functioninterfaces.FunctionHandle, error) {
			return catalog.LookupBuiltInFunction("text_equal", []types.Type{left, right})
		},
		func() (functioninterfaces.FunctionHandle, error) {
			return catalog.ComparisonFunction(foundation.Equal, left, right)
		},
		func() (functioninterfaces.FunctionHandle, error) {
			return catalog.ComparisonFunction(foundation.LessThan, left, left)
		},
	} {
		handle, err := resolve()
		require.ErrorIs(t, err, shared.ErrUnsupportedLookup)
		require.Nil(t, handle)
	}
}

func TestDeclarationCanonicalizationAndAmbiguity(t *testing.T) {
	def := textDeclarations(t)[0]
	builder := NewFunctionCatalogBuilder()
	require.NoError(t, builder.RegisterDeclaration(def))
	renamed := def
	x, err := types.NewVarCharTypePattern("renamed", foundation.NonNull)
	require.NoError(t, err)
	renamed.Signature, err = shared.NewDeclarationSignature([]types.TypePattern{x, x}, def.Signature.Result())
	require.NoError(t, err)
	require.NoError(t, builder.RegisterDeclaration(renamed))
	conflict := renamed
	conflict.NullPolicies = []shared.NullPolicy{shared.NullPolicyUnspecified, shared.PropagateNull}
	require.ErrorIs(t, builder.RegisterDeclaration(conflict), merr.ErrServiceInternal)
	// Caller-owned policies cannot mutate the published contract.
	def.NullPolicies[0] = shared.NullPolicyUnspecified
	catalog, err := builder.Build()
	require.NoError(t, err)
	require.Len(t, catalog.entries, 1)
	size := uint64(32)
	text := textType(t, &size, foundation.MayNull)
	_, err = catalog.ComparisonFunction(foundation.Equal, text, text)
	require.NoError(t, err)

	// A fixed specialization overlaps the parameterized declaration; neither wins by order.
	def = textDeclarations(t)[0]
	concrete := textType(t, &size, foundation.NonNull)
	metadata, err := def.bind([]types.Type{concrete, concrete})
	require.NoError(t, err)
	fixed := FunctionDefinition{Name: def.Name, Metadata: metadata}
	for _, fixedFirst := range []bool{false, true} {
		builder := NewFunctionCatalogBuilder()
		if fixedFirst {
			require.NoError(t, builder.Register(fixed))
		}
		require.NoError(t, builder.RegisterDeclaration(def))
		if !fixedFirst {
			require.NoError(t, builder.Register(fixed))
		}
		catalog, err := builder.Build()
		require.NoError(t, err)
		_, err = catalog.LookupFunction(def.Name, []types.Type{concrete, concrete})
		require.ErrorIs(t, err, shared.ErrUnsupportedLookup)
		_, err = catalog.LookupBuiltInFunction("text_equal", []types.Type{concrete, concrete})
		require.ErrorIs(t, err, shared.ErrUnsupportedLookup)
		_, err = catalog.ComparisonFunction(foundation.Equal, concrete, concrete)
		require.ErrorIs(t, err, shared.ErrUnsupportedLookup)
	}
}

func TestParameterizedCallbackUsesBoundMetadata(t *testing.T) {
	definitions := textDeclarations(t)
	catalog := declarationCatalog(t, definitions...)
	small, large := uint64(4), uint64(64)
	sample := textType(t, &small, foundation.NonNull)
	handle, err := catalog.ComparisonFunction(foundation.Equal, sample, sample)
	require.NoError(t, err)
	builder := NewFunctionContextBuilder(catalog)
	calls := 0
	require.NoError(t, builder.BindImplementation(handle, func(args []valueinterfaces.Value) (valueinterfaces.Value, error) {
		calls++
		require.Equal(t, foundation.NonNull, args[0].Type().Nullability())
		require.True(t, args[0].Type().EqualsTo(args[1].Type()))
		boolean, err := types.NewBoolType(foundation.NonNull)
		require.NoError(t, err)
		return values.Factory.NewBool(boolean, args[0].(valueinterfaces.VarCharValue).VarChar() == args[1].(valueinterfaces.VarCharValue).VarChar())
	}))
	context, err := builder.Build()
	require.NoError(t, err)
	for _, size := range []uint64{small, large} {
		nullable := textType(t, &size, foundation.MayNull)
		concrete := textType(t, &size, foundation.NonNull)
		resolved, err := context.ComparisonFunction(foundation.Equal, nullable, concrete)
		require.NoError(t, err)
		metadata, err := context.Metadata(resolved)
		require.NoError(t, err)
		signature, err := shared.BindFunctionSignature(metadata, []types.Type{nullable, concrete})
		require.NoError(t, err)
		left, err := values.Factory.NewVarChar(nullable, "猫")
		require.NoError(t, err)
		right, err := values.Factory.NewVarChar(concrete, "猫")
		require.NoError(t, err)
		result, err := context.Evaluate(resolved, signature, []valueinterfaces.Value{left, right})
		require.NoError(t, err)
		require.True(t, result.(valueinterfaces.BoolValue).Bool())
		require.Equal(t, foundation.MayNull, result.Type().Nullability())
		null, err := values.Factory.NewNull(nullable)
		require.NoError(t, err)
		result, err = context.Evaluate(resolved, signature, []valueinterfaces.Value{null, right})
		require.NoError(t, err)
		require.True(t, result.IsNull())
		// A signature from another bound cannot reach the callback.
		_, err = context.Evaluate(resolved, handle.Signature(), []valueinterfaces.Value{left, right})
		require.ErrorIs(t, err, shared.ErrContractViolation)
		ne, err := context.ComparisonFunction(foundation.NotEqual, concrete, concrete)
		require.NoError(t, err)
		_, err = context.Evaluate(ne, ne.Signature(), []valueinterfaces.Value{right, right})
		require.ErrorIs(t, err, shared.ErrEvaluationUnavailable)
	}
	require.Equal(t, 2, calls)
}

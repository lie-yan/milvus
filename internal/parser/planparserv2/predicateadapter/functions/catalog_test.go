package functions

import (
	"testing"

	"github.com/lie-yan/predicate-optimization/foundation"
	shared "github.com/lie-yan/predicate-optimization/functions"
	"github.com/lie-yan/predicate-optimization/types"
	"github.com/stretchr/testify/require"

	"github.com/milvus-io/milvus/pkg/v3/util/merr"
)

func registration(t *testing.T, name string, op foundation.OperatorType, builtin shared.BuiltInFunctionName, policy shared.NullPolicy) FunctionDefinition {
	t.Helper()
	integer, err := types.NewIntType(types.Int64, foundation.NonNull)
	require.NoError(t, err)
	boolean, err := types.NewBoolType(foundation.NonNull)
	require.NoError(t, err)
	args, result := []types.Type{integer, integer}, types.Type(boolean)
	var operator foundation.Optional[foundation.OperatorType]
	var builtInName foundation.Optional[shared.BuiltInFunctionName]
	if op != 0 {
		operator = foundation.Some(op)
	}
	if op.IsArithmeticOperator() {
		result = integer
	}
	if builtin != "" {
		builtInName = foundation.Some(builtin)
	}
	sig, err := shared.NewFunctionSignature(args, result)
	require.NoError(t, err)
	metadata, err := shared.NewFunctionMetadata(sig, name, builtInName, operator, []shared.NullPolicy{policy, policy})
	require.NoError(t, err)
	return FunctionDefinition{Name: FunctionName{Namespace: "test.scalar", Name: name}, Metadata: metadata}
}

func TestCatalogRegistrationIdentity(t *testing.T) {
	builder := NewFunctionCatalogBuilder()
	less := registration(t, "less", foundation.LessThan, "", shared.PropagateNull)
	add := registration(t, "add", foundation.Add, "", shared.PropagateNull)
	ordinary := registration(t, "ordinary", 0, "ordinary", shared.PropagateNull)
	for _, def := range []FunctionDefinition{less, add, ordinary} {
		require.NoError(t, builder.Register(def))
	}
	// Fresh but equal Types and a different display label must retain the first record.
	duplicate := registration(t, "different label", foundation.LessThan, "", shared.PropagateNull)
	duplicate.Name = less.Name
	require.NoError(t, builder.Register(duplicate))

	// Same name and arguments with a different result remain distinct registrations.
	otherResult := ordinary
	integer, err := types.NewIntType(types.Int64, foundation.NonNull)
	require.NoError(t, err)
	sig, err := shared.NewFunctionSignature(ordinary.Metadata.Signature().ArgumentTypes(), integer)
	require.NoError(t, err)
	otherResult.Metadata, err = shared.NewFunctionMetadata(sig, "ordinary", ordinary.Metadata.BuiltInName(), ordinary.Metadata.Operator(), ordinary.Metadata.NullPolicies())
	require.NoError(t, err)
	require.NoError(t, builder.Register(otherResult))
	require.NoError(t, builder.Register(otherResult))

	otherNamespace := less
	otherNamespace.Name.Namespace = "another.namespace"
	require.NoError(t, builder.Register(otherNamespace))

	catalog, err := builder.Build()
	require.NoError(t, err)
	require.Len(t, catalog.entries, 5)
	require.Same(t, less.Metadata, catalog.entries[0].metadata)
	require.Same(t, add.Metadata, catalog.entries[1].metadata)
	require.Same(t, ordinary.Metadata, catalog.entries[2].metadata)
	require.Same(t, otherResult.Metadata, catalog.entries[3].metadata)
	require.Equal(t, otherNamespace.Name, catalog.entries[4].name)
}

func TestCatalogRejectsConflictsWithoutMutation(t *testing.T) {
	original := registration(t, "less", foundation.LessThan, "", shared.PropagateNull)
	for _, tc := range []struct {
		name string
		def  FunctionDefinition
	}{
		{"operator", registration(t, "less", foundation.GreaterThan, "", shared.PropagateNull)},
		{"builtin", registration(t, "less", foundation.LessThan, "alias", shared.PropagateNull)},
		{"null policy", registration(t, "less", foundation.LessThan, "", shared.NullPolicyUnspecified)},
	} {
		t.Run(tc.name, func(t *testing.T) {
			builder := NewFunctionCatalogBuilder()
			require.NoError(t, builder.Register(original))
			require.ErrorIs(t, builder.Register(tc.def), merr.ErrServiceInternal)
			// Failed registration leaves the builder usable and the existing entry intact.
			require.NoError(t, builder.Register(registration(t, "add", foundation.Add, "", shared.PropagateNull)))
			catalog, err := builder.Build()
			require.NoError(t, err)
			require.Len(t, catalog.entries, 2)
			require.Same(t, original.Metadata, catalog.entries[0].metadata)
		})
	}
}

func TestCatalogBuilderOwnershipAndLifecycle(t *testing.T) {
	def := registration(t, "less", foundation.LessThan, "", shared.PropagateNull)
	original := def
	builder := NewFunctionCatalogBuilder()
	copyOfBuilder := *builder
	require.NoError(t, builder.Register(def))
	def.Name.Name = "changed"
	def.Metadata = nil
	catalog, err := copyOfBuilder.Build()
	require.NoError(t, err)
	require.Equal(t, original.Name, catalog.entries[0].name)
	require.Same(t, original.Metadata, catalog.entries[0].metadata)
	for _, consumed := range []*FunctionCatalogBuilder{builder, &copyOfBuilder} {
		require.ErrorIs(t, consumed.Register(original), merr.ErrServiceInternal)
		result, err := consumed.Build()
		require.ErrorIs(t, err, merr.ErrServiceInternal)
		require.Nil(t, result)
	}
}

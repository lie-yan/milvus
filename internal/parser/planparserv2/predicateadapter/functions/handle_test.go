package functions

import (
	"testing"

	"github.com/lie-yan/predicate-optimization/foundation"
	shared "github.com/lie-yan/predicate-optimization/functions"
	functioninterfaces "github.com/lie-yan/predicate-optimization/functions/interfaces"
	bundled "github.com/lie-yan/predicate-optimization/testbed/functions"
	"github.com/stretchr/testify/require"
)

func catalogWith(t *testing.T, definitions ...FunctionDefinition) *FunctionCatalog {
	t.Helper()
	builder := NewFunctionCatalogBuilder()
	for _, def := range definitions {
		require.NoError(t, builder.Register(def))
	}
	catalog, err := builder.Build()
	require.NoError(t, err)
	return catalog
}

func TestFunctionHandleIdentityAndMetadata(t *testing.T) {
	less := registration(t, "less", foundation.LessThan, "", shared.PropagateNull)
	add := registration(t, "add", foundation.Add, "", shared.PropagateNull)
	catalog := catalogWith(t, less, add)
	first, second := catalog.handleAt(0), catalog.handleAt(0)
	require.Same(t, first, second)
	require.True(t, first.EqualsTo(second))
	require.False(t, first.EqualsTo(catalog.handleAt(1)))
	metadata, err := catalog.Metadata(second)
	require.NoError(t, err)
	require.Same(t, less.Metadata, metadata)
	require.Same(t, metadata.Signature(), first.Signature())
}

func TestFunctionHandleOwnershipAndFailures(t *testing.T) {
	def := registration(t, "less", foundation.LessThan, "", shared.PropagateNull)
	catalog, foreign := catalogWith(t, def), catalogWith(t, def)
	handle, foreignHandle := catalog.handleAt(0), foreign.handleAt(0)
	require.False(t, handle.EqualsTo(foreignHandle))
	peer, err := bundled.NewCatalog([]bundled.Definition{{Namespace: def.Name.Namespace, Name: def.Name.Name, Metadata: def.Metadata}})
	require.NoError(t, err)
	peerHandle := peer.Handles()[0]
	for _, tc := range []struct {
		name   string
		handle functioninterfaces.FunctionHandle
		want   error
	}{
		{"foreign snapshot", foreignHandle, shared.ErrIncompatibleHandle},
		{"foreign representation", peerHandle, shared.ErrIncompatibleHandle},
		{"missing entry", &functionHandle{catalog: catalog, index: len(catalog.entries)}, shared.ErrUnknownHandle},
	} {
		t.Run(tc.name, func(t *testing.T) {
			metadata, err := catalog.Metadata(tc.handle)
			require.ErrorIs(t, err, tc.want)
			require.Nil(t, metadata)
		})
	}
}

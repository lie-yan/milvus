package utils

import (
	"strconv"
	"testing"

	expressioninterfaces "github.com/lie-yan/predicate-optimization/expressions/interfaces"
	"github.com/lie-yan/predicate-optimization/foundation"
	"github.com/lie-yan/predicate-optimization/types"
	"github.com/stretchr/testify/require"

	adapterexpressions "github.com/milvus-io/milvus/internal/parser/planparserv2/predicateadapter/expressions"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
)

// Compatible key representation that cannot be used as a Go map key.
type preservedID struct{ parts []string }

func (id preservedID) Key() string    { return id.parts[0] }
func (id preservedID) String() string { return "opaque peer" }
func (id preservedID) EqualsTo(other expressioninterfaces.OpaqueID) bool {
	return id.Key() == other.Key()
}

func TestOpaqueResolutionRejectsForeignOrChangedIdentity(t *testing.T) {
	registry, err := NewOpaqueRegistry()
	require.NoError(t, err)
	foreign, err := NewOpaqueRegistry()
	require.NoError(t, err)
	boolean, err := types.NewBoolType(foundation.NonNull)
	require.NoError(t, err)
	nullable, err := types.NewBoolType(foundation.MayNull)
	require.NoError(t, err)
	integer, err := types.NewIntType(types.Int64, foundation.NonNull)
	require.NoError(t, err)
	binding, err := registry.Register(7, boolean)
	require.NoError(t, err)
	other, err := foreign.Register(7, boolean)
	require.NoError(t, err)
	for _, id := range []expressioninterfaces.OpaqueID{
		other.id,
		adapterexpressions.NewOpaqueID(uint64(registry.scope), 1),
		preservedID{[]string{registry.prefix + "00"}},
		preservedID{[]string{registry.prefix + "invalid"}},
	} {
		_, err := registry.Resolve(id, boolean, 7)
		require.ErrorIs(t, err, merr.ErrServiceInternal)
	}
	_, err = registry.Resolve(preservedID{[]string{registry.prefix + "invalid"}}, boolean, 7)
	var parseError *strconv.NumError
	require.ErrorAs(t, err, &parseError)
	for _, changed := range []types.Type{nullable, integer} {
		_, err := registry.Resolve(binding.id, changed, 7)
		require.ErrorIs(t, err, merr.ErrServiceInternal)
	}
	require.Len(t, registry.bindings, 1) // Failed resolution never registers a region.
}

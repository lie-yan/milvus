package schema

import (
	"testing"

	expressioninterfaces "github.com/lie-yan/predicate-optimization/expressions/interfaces"
	"github.com/lie-yan/predicate-optimization/foundation"
	"github.com/lie-yan/predicate-optimization/types"
	"github.com/stretchr/testify/require"
	"google.golang.org/protobuf/proto"

	adapterexpressions "github.com/milvus-io/milvus/internal/parser/planparserv2/predicateadapter/expressions"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
)

// Independent ID representation implementing the same canonical-key contract.
type registryIDPeer []string

func (id registryIDPeer) Key() string    { return id[0] }
func (id registryIDPeer) String() string { return "diagnostic name" }
func (id registryIDPeer) EqualsTo(other expressioninterfaces.ColumnID) bool {
	return id.Key() == other.Key()
}

func TestFieldRegistryBinding(t *testing.T) {
	int64NonNullType, _ := types.NewIntType(types.Int64, foundation.NonNull)
	int64MayNullType, _ := types.NewIntType(types.Int64, foundation.MayNull)
	boolNonNullType, _ := types.NewBoolType(foundation.NonNull)
	int8NonNullType, _ := types.NewIntType(types.Int8, foundation.NonNull)
	schema, _, view := schemaFixture(t)
	registry, err := NewFieldRegistry(view)
	require.NoError(t, err)
	info := columnFixture(schema.Fields[0])
	before := proto.Clone(info)
	first, supported, err := registry.BindColumn(info, int64NonNullType)
	require.NoError(t, err)
	require.True(t, supported)
	equivalentType, err := types.NewIntType(types.Int64, foundation.NonNull)
	require.NoError(t, err)
	repeated, supported, err := registry.BindColumn(columnFixture(schema.Fields[0]), equivalentType)
	require.NoError(t, err)
	require.True(t, supported)
	require.Same(t, first.ColumnID(), repeated.ColumnID())
	require.Same(t, first.Type(), repeated.Type())
	resolved, err := registry.ResolveColumn(registryIDPeer{first.ColumnID().Key()}, equivalentType)
	require.NoError(t, err)
	require.Same(t, schema.Fields[0], resolved.Field())
	require.True(t, proto.Equal(before, info))
	// Binding consistency is independent of the motivating example's domain.
	for _, tc := range []struct {
		index int
		typ   types.Type
	}{
		{1, boolNonNullType}, {2, int64MayNullType}, {3, int8NonNullType},
	} {
		binding, supported, err := registry.BindColumn(columnFixture(schema.Fields[tc.index]), tc.typ)
		require.NoError(t, err)
		require.True(t, supported)
		require.False(t, first.ColumnID().EqualsTo(binding.ColumnID()))
		resolved, err := registry.ResolveColumn(binding.ColumnID(), tc.typ)
		require.NoError(t, err)
		require.Same(t, schema.Fields[tc.index], resolved.Field())
	}
}

func TestFieldRegistryBindingConflicts(t *testing.T) {
	int64NonNullType, _ := types.NewIntType(types.Int64, foundation.NonNull)
	int64MayNullType, _ := types.NewIntType(types.Int64, foundation.MayNull)
	boolNonNullType, _ := types.NewBoolType(foundation.NonNull)
	schema, _, view := schemaFixture(t)
	registry, err := NewFieldRegistry(view)
	require.NoError(t, err)
	info := columnFixture(schema.Fields[0])
	first, _, err := registry.BindColumn(info, int64NonNullType)
	require.NoError(t, err)
	_, _, err = registry.BindColumn(info, boolNonNullType)
	require.ErrorIs(t, err, merr.ErrServiceInternal)
	_, _, err = registry.BindColumn(info, int64MayNullType)
	require.ErrorIs(t, err, merr.ErrServiceInternal)
	conflicting := columnFixture(schema.Fields[0])
	conflicting.IsAutoID = false
	_, _, err = registry.BindColumn(conflicting, int64NonNullType)
	require.ErrorIs(t, err, merr.ErrServiceInternal)
	// Failures leave the original binding available for resolution.
	resolved, err := registry.ResolveColumn(first.ColumnID(), int64NonNullType)
	require.NoError(t, err)
	require.Equal(t, first, resolved)
	sub := schema.StructArrayFields[0].Fields[0]
	_, supported, err := registry.BindColumn(columnFixture(sub), int64NonNullType)
	require.NoError(t, err)
	require.False(t, supported)
	_, err = registry.ResolveColumn(adapterexpressions.NewColumnID(uint64(registry.token), sub.GetFieldID()), int64NonNullType)
	require.ErrorIs(t, err, merr.ErrServiceInternal)
}

func TestFieldRegistryResolution(t *testing.T) {
	int64NonNullType, _ := types.NewIntType(types.Int64, foundation.NonNull)
	int64MayNullType, _ := types.NewIntType(types.Int64, foundation.MayNull)
	boolNonNullType, _ := types.NewBoolType(foundation.NonNull)
	schema, _, view := schemaFixture(t)
	first, err := NewFieldRegistry(view)
	require.NoError(t, err)
	second, err := NewFieldRegistry(view)
	require.NoError(t, err)
	info := columnFixture(schema.Fields[0])
	a, _, err := first.BindColumn(info, int64NonNullType)
	require.NoError(t, err)
	b, _, err := second.BindColumn(info, int64NonNullType)
	require.NoError(t, err)
	require.False(t, a.ColumnID().EqualsTo(b.ColumnID()))
	for name, id := range map[string]expressioninterfaces.ColumnID{
		"foreign session":    b.ColumnID(),
		"unregistered field": adapterexpressions.NewColumnID(uint64(first.token), 102),
		"noncanonical field": registryIDPeer{first.prefix + "0101"},
		"invalid field":      registryIDPeer{first.prefix + "not-a-field"},
	} {
		t.Run(name, func(t *testing.T) {
			_, err := first.ResolveColumn(id, int64NonNullType)
			require.ErrorIs(t, err, merr.ErrServiceInternal)
		})
	}
	_, err = first.ResolveColumn(a.ColumnID(), boolNonNullType)
	require.ErrorIs(t, err, merr.ErrServiceInternal)
	_, err = first.ResolveColumn(a.ColumnID(), int64MayNullType)
	require.ErrorIs(t, err, merr.ErrServiceInternal)
}

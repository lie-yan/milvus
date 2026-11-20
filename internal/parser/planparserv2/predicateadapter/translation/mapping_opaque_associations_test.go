package translation

import (
	"testing"

	"github.com/lie-yan/predicate-optimization/expressions"
	expressioninterfaces "github.com/lie-yan/predicate-optimization/expressions/interfaces"
	"github.com/lie-yan/predicate-optimization/foundation"
	"github.com/lie-yan/predicate-optimization/types"
	"github.com/stretchr/testify/require"

	translationutils "github.com/milvus-io/milvus/internal/parser/planparserv2/predicateadapter/translation/utils"
	"github.com/milvus-io/milvus/pkg/v3/proto/planpb"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
)

// Compatible key representation that cannot be used as a Go map key.
type preservedID struct{ parts []string }

func (id preservedID) Key() string    { return id.parts[0] }
func (id preservedID) String() string { return "opaque peer" }
func (id preservedID) EqualsTo(other expressioninterfaces.OpaqueID) bool {
	return id.Key() == other.Key()
}

func TestOpaqueOccurrenceAssociations(t *testing.T) {
	registry, err := translationutils.NewOpaqueRegistry()
	require.NoError(t, err)
	boolean, err := types.NewBoolType(foundation.NonNull)
	require.NoError(t, err)
	source := &planpb.Expr{Expr: &planpb.Expr_NullExpr{NullExpr: &planpb.NullExpr{Op: planpb.NullExpr_IsNull, ColumnInfo: &planpb.ColumnInfo{FieldId: 101}}}}
	var occurrences []occurrence
	var bindings []translationutils.OpaqueBinding
	for range 2 {
		// Reserve the next source position before constructing its graph leaf.
		binding, err := registry.Register(len(occurrences), boolean)
		require.NoError(t, err)
		graph, err := expressions.NewOpaqueExpression(binding.ID(), binding.Type(), "null predicate")
		require.NoError(t, err)
		id := appendOccurrence(&occurrences, source, graph, nil)
		require.Equal(t, binding.Occurrence(), int(id))
		bindings = append(bindings, binding)
	}
	require.False(t, bindings[0].ID().EqualsTo(bindings[1].ID()))
	for _, binding := range bindings {
		// Rebuilt graph objects and compatible external identities retain the association.
		graph, err := expressions.NewOpaqueExpression(preservedID{[]string{binding.ID().Key()}}, boolean, "rebuilt")
		require.NoError(t, err)
		resolved, err := registry.Resolve(graph.ID(), graph.Type(), binding.Occurrence())
		require.NoError(t, err)
		require.Same(t, source, occurrences[resolved.Occurrence()].source)
		require.Same(t, boolean, resolved.Type())

	}
	_, err = registry.Resolve(bindings[1].ID(), boolean, bindings[0].Occurrence())
	require.ErrorIs(t, err, merr.ErrServiceInternal)
}

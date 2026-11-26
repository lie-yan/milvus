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
)

// Compatible key representation that cannot be used as a Go map key.
type preservedID struct{ parts []string }

func (id preservedID) Key() string    { return id.parts[0] }
func (id preservedID) String() string { return "opaque peer" }
func (id preservedID) EqualsTo(other expressioninterfaces.OpaqueID) bool {
	return id.Key() == other.Key()
}

func TestOpaqueSourceBindings(t *testing.T) {
	registry, err := translationutils.NewOpaqueRegistry()
	require.NoError(t, err)
	boolean, err := types.NewBoolType(foundation.NonNull)
	require.NoError(t, err)
	source := &planpb.Expr{Expr: &planpb.Expr_NullExpr{NullExpr: &planpb.NullExpr{Op: planpb.NullExpr_IsNull, ColumnInfo: &planpb.ColumnInfo{FieldId: 101}}}}

	var bindings []translationutils.OpaqueBinding
	for range 2 {
		binding, err := registry.Register(source, boolean)
		require.NoError(t, err)
		bindings = append(bindings, binding)
	}
	require.False(t, bindings[0].ID().EqualsTo(bindings[1].ID()))
	for _, binding := range bindings {
		graph, err := expressions.NewOpaqueExpression(preservedID{[]string{binding.ID().Key()}}, boolean, "rebuilt")
		require.NoError(t, err)
		resolved, err := registry.Resolve(graph.ID(), graph.Type())
		require.NoError(t, err)
		require.Same(t, source, resolved.Source())
		// Export needs no occurrence records, even for a rebuilt compatible identity.
		exporter := exporter{imported: &ImportResult{opaques: registry}}
		output, err := exporter.exportOpaque(graph)
		require.NoError(t, err)
		output.GetNullExpr().ColumnInfo.FieldId = 999
		require.Equal(t, int64(101), source.GetNullExpr().ColumnInfo.FieldId)
	}
}

func TestImportWithoutOpaqueRegistry(t *testing.T) {
	translator, _, columns := operandFixture(t)
	source := &planpb.Expr{Expr: &planpb.Expr_AlwaysTrueExpr{AlwaysTrueExpr: &planpb.AlwaysTrueExpr{}}}
	imported, err := translator.Import(t.Context(), source)
	require.NoError(t, err)
	require.Nil(t, imported.opaques)
	output, err := translator.Export(t.Context(), imported, imported.Root())
	require.NoError(t, err)
	require.NotNil(t, output.GetAlwaysTrueExpr())
	opaqueSource := &planpb.Expr{Expr: &planpb.Expr_NullExpr{NullExpr: &planpb.NullExpr{ColumnInfo: columns[3], Op: planpb.NullExpr_IsNull}}}
	opaque, err := translator.Import(t.Context(), opaqueSource)
	require.NoError(t, err)
	require.NotNil(t, opaque.opaques)
	output, err = translator.Export(t.Context(), imported, opaque.Root())
	require.Nil(t, output)
	requireOperandFailure(t, err, TranslationDefect, StageExport)
}

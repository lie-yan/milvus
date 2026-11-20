package translation

import (
	"testing"

	"github.com/lie-yan/predicate-optimization/expressions"
	"github.com/lie-yan/predicate-optimization/foundation"
	"github.com/lie-yan/predicate-optimization/types"
	"github.com/stretchr/testify/require"

	"github.com/milvus-io/milvus/internal/parser/planparserv2/predicateadapter/values"
	"github.com/milvus-io/milvus/pkg/v3/proto/planpb"
)

func TestOccurrenceIdentityAndChildOwnership(t *testing.T) {
	boolean, err := types.NewBoolType(foundation.NonNull)
	require.NoError(t, err)
	value, err := values.Factory.NewBool(boolean, true)
	require.NoError(t, err)
	graph, err := expressions.NewLiteralExpression(value)
	require.NoError(t, err)
	source := &planpb.Expr{Expr: &planpb.Expr_AlwaysTrueExpr{AlwaysTrueExpr: &planpb.AlwaysTrueExpr{}}}
	var table []occurrence
	first := appendOccurrence(&table, source, graph, nil)
	second := appendOccurrence(&table, source, graph, nil)
	require.NotEqual(t, first, second)
	children := []occurrenceID{first, second}
	parent := appendOccurrence(&table, &planpb.Expr{}, graph, children)
	children[0] = second
	require.Equal(t, []occurrenceID{first, second}, table[parent].children)
	require.Same(t, source, table[first].source)
	require.Same(t, graph, table[second].imported)
}

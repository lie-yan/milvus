package translation

import (
	"context"

	"github.com/lie-yan/predicate-optimization/expressions"

	"github.com/milvus-io/milvus/internal/parser/planparserv2/predicateadapter/internal/assertions"
	"github.com/milvus-io/milvus/pkg/v3/proto/planpb"
)

// Per-export reconstruction state with no ownership of borrowed source metadata.
type exporter struct {
	// Request context for this invocation.
	ctx context.Context
	// Immutable graph/source and preservation associations.
	imported *ImportResult
}

// Dispatches actual checked graph nodes to their paired reconstruction handlers.
func (v *exporter) exportExpr(id occurrenceID, graph expressions.Expression) (*planpb.Expr, error) {
	if err := v.ctx.Err(); err != nil {
		return nil, err
	}
	if v.imported.occurrences[id].IsBinaryRange() {
		return v.exportBinaryRange(id, graph)
	}
	if v.imported.occurrences[id].IsTerm() {
		return v.exportTerm(id, graph)
	}
	switch node := graph.(type) {
	case *expressions.LiteralExpression, *expressions.LogicalExpression:
		return v.exportLogical(id, node)
	case *expressions.CallExpression:
		if v.imported.occurrences[id].IsFieldComparison() {
			return v.exportCompare(id, node)
		}
		assertions.True(v.imported.occurrences[id].IsUnaryComparison())
		return v.exportUnaryRange(id, node)
	case *expressions.OpaqueExpression:
		return v.exportOpaque(id, node)
	default:
		return nil, translationDefect(StageExport, "graph has no paired export route")
	}
}

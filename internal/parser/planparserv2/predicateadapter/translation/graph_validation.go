package translation

import (
	"context"

	"github.com/lie-yan/predicate-optimization/expressions"

	"github.com/milvus-io/milvus/internal/parser/planparserv2/predicateadapter/internal/assertions"
)

// Read-only adapter validation following framework graph validation.
type graphChecker struct {
	// Request context for this invocation.
	ctx context.Context
	// Immutable import associations.
	imported *ImportResult
	// Stage requesting the graph checks.
	stage Stage
}

// Checks the actual node variant against its source-backed route.
func (v *graphChecker) checkExpr(id occurrenceID, graph expressions.Expression) error {
	if err := v.ctx.Err(); err != nil {
		return err
	}
	if id < 0 || int(id) >= len(v.imported.occurrences) {
		return translationDefect(v.stage, "missing graph occurrence")
	}
	if v.imported.occurrences[id].IsBinaryRange() {
		return v.checkBinaryRange(id, graph)
	}
	if v.imported.occurrences[id].IsTerm() {
		return v.checkTerm(id, graph)
	}
	switch node := graph.(type) {
	case *expressions.LiteralExpression, *expressions.LogicalExpression:
		return v.checkLogical(id, node)
	case *expressions.CallExpression:
		record := v.imported.occurrences[id]
		if record.IsFieldComparison() {
			return v.checkCompare(id, node)
		}
		// A candidate call can replace a non-call source occurrence; reject it before asserting internal routing.
		if !record.IsUnaryComparison() {
			return translationDefect(v.stage, "call has no paired comparison route")
		}
		assertions.True(record.IsUnaryComparison())
		return v.checkUnaryRange(id, node)
	case *expressions.OpaqueExpression:
		_, err := resolveOpaque(v.imported, id, node, v.stage)
		return err
	default:
		return translationDefect(v.stage, "graph has no paired route")
	}
}

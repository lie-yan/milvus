package translation

import (
	"context"

	"github.com/lie-yan/predicate-optimization/expressions"

	"github.com/milvus-io/milvus/internal/parser/planparserv2/predicateadapter/internal/assertions"
	"github.com/milvus-io/milvus/pkg/v3/proto/planpb"
)

// Independent emitted-protobuf inspection using the admitted source contracts.
type candidateChecker struct {
	// Request context for this invocation.
	ctx context.Context
	// Immutable source and preservation associations.
	imported *ImportResult
}

// Selects the expected paired route from its occurrence, then inspects actual output.
func (v *candidateChecker) checkExpr(id occurrenceID, candidate *planpb.Expr) error {
	if err := v.ctx.Err(); err != nil {
		return err
	}
	if id < 0 || int(id) >= len(v.imported.occurrences) {
		return translationDefect(StageCandidateValidation, "missing candidate occurrence")
	}
	record := v.imported.occurrences[id]
	if record.IsBinaryRange() {
		return v.checkBinaryRange(record, candidate)
	}
	if record.IsTerm() {
		return v.checkTerm(record, candidate)
	}
	switch record.imported.(type) {
	case *expressions.LiteralExpression, *expressions.LogicalExpression:
		return v.checkLogical(record, candidate)
	case *expressions.CallExpression:
		if record.IsFieldComparison() {
			return v.checkCompare(record, candidate)
		}
		assertions.True(record.IsUnaryComparison())
		return v.checkUnaryRange(record, candidate)
	case *expressions.OpaqueExpression:
		return v.checkOpaque(record, candidate)
	default:
		return translationDefect(StageCandidateValidation, "missing paired candidate route")
	}
}

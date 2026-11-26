package translation

import (
	"context"

	"github.com/lie-yan/predicate-optimization/expressions"
	"github.com/lie-yan/predicate-optimization/foundation"
	valueinterfaces "github.com/lie-yan/predicate-optimization/values/interfaces"

	"github.com/milvus-io/milvus/pkg/v3/proto/planpb"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
)

// Per-export state; borrowed bindings remain immutable.
type exporter struct {
	// Cancellation for this invocation.
	ctx context.Context
	// Immutable field and opaque bindings owned by this attempt.
	imported *ImportResult
}

// Dispatches the actual graph without consulting source occurrences.
func (v *exporter) exportExpr(graph expressions.Expression) (*planpb.Expr, error) {
	if err := v.ctx.Err(); err != nil {
		return nil, err
	}
	if graph == nil {
		return nil, translationDefect(StageExport, "missing expression")
	}
	switch node := graph.(type) {
	case *expressions.OpaqueExpression:
		return v.exportOpaque(node)
	case *expressions.LiteralExpression:
		value, ok := node.Value().(valueinterfaces.BoolValue)
		if !ok || node.Type().Nullability() != foundation.NonNull || value.IsNull() {
			return nil, unsupportedExport("expected nonnull Boolean predicate")
		}
		result := &planpb.Expr{Expr: &planpb.Expr_AlwaysTrueExpr{AlwaysTrueExpr: &planpb.AlwaysTrueExpr{}}}
		if !value.Bool() {
			result = &planpb.Expr{Expr: &planpb.Expr_UnaryExpr{UnaryExpr: &planpb.UnaryExpr{Op: planpb.UnaryExpr_Not, Child: result}}}
		}
		return result, nil
	case *expressions.LogicalExpression:
		return v.exportLogical(node)
	case *expressions.CallExpression:
		return v.exportComparison(node)
	case *expressions.SpecialFormExpression:
		return v.exportTerm(node)
	default:
		return nil, unsupportedExport("unrepresentable predicate expression")
	}
}

// Reports a valid graph outside executable export coverage, without a partial candidate.
func unsupportedExport(message string) error {
	return newFailure(UnsupportedExpression, StageExport, merr.Wrap(merr.ErrServiceUnimplemented, "predicate translation: "+message))
}

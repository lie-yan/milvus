package translation

import (
	"context"
	"errors"

	"github.com/lie-yan/predicate-optimization/expressions"
	shared "github.com/lie-yan/predicate-optimization/functions"
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
		// A nullable declared type can still carry a known nonnull constant.
		if !ok || value.IsNull() {
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
		return v.exportCall(node)
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

// Keeps comparison export on its existing lookup path and resolves other owned built-ins on a miss.
func (v *exporter) exportCall(call *expressions.CallExpression) (*planpb.Expr, error) {
	op, err := v.imported.owner.runtime.ComparisonOperator(call.Handle())
	if err == nil {
		return v.exportComparison(call, op)
	}
	if !errors.Is(err, shared.ErrUnsupportedLookup) {
		return nil, newFailure(TranslationDefect, StageExport, err)
	}
	metadata, err := v.imported.owner.context.Metadata(call.Handle())
	if err != nil {
		return nil, newFailure(TranslationDefect, StageExport, err)
	}
	if name, present := metadata.BuiltInName().Get(); present && (name == shared.BuiltInArrayContains || name == shared.BuiltInArrayContainsAny) {
		return v.exportArrayMembership(call, name)
	}
	return nil, unsupportedExport("function has no executable mapping")
}

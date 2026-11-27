package translation

import (
	"github.com/lie-yan/predicate-optimization/expressions"
	"github.com/lie-yan/predicate-optimization/foundation"

	"github.com/milvus-io/milvus/pkg/v3/proto/planpb"
)

// Lowers n-ary logic into an order-preserving balanced native tree.
func (v *exporter) exportLogical(graph *expressions.LogicalExpression) (*planpb.Expr, error) {
	if graph.Operator() == foundation.Not {
		child, err := v.exportExpr(graph.ChildAt(0))
		if err != nil {
			return nil, err
		}
		return &planpb.Expr{Expr: &planpb.Expr_UnaryExpr{UnaryExpr: &planpb.UnaryExpr{Op: planpb.UnaryExpr_Not, Child: child}}}, nil
	}
	op := planpb.BinaryExpr_LogicalAnd
	switch graph.Operator() {
	case foundation.And:
	case foundation.Or:
		op = planpb.BinaryExpr_LogicalOr
	default:
		return nil, unsupportedExport("unmapped logical operator")
	}
	return v.exportLogicalRange(graph, op, 0, graph.ChildCount())
}

// Recursively splits child indexes without copying the graph's child slice.
func (v *exporter) exportLogicalRange(graph *expressions.LogicalExpression, op planpb.BinaryExpr_BinaryOp, begin, end int) (*planpb.Expr, error) {
	if end-begin == 1 {
		return v.exportExpr(graph.ChildAt(begin))
	}
	middle := begin + (end-begin)/2
	left, err := v.exportLogicalRange(graph, op, begin, middle)
	if err != nil {
		return nil, err
	}
	right, err := v.exportLogicalRange(graph, op, middle, end)
	if err != nil {
		return nil, err
	}
	return &planpb.Expr{Expr: &planpb.Expr_BinaryExpr{BinaryExpr: &planpb.BinaryExpr{Op: op, Left: left, Right: right}}}, nil
}

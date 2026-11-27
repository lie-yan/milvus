package translation

import (
	"github.com/lie-yan/predicate-optimization/expressions"

	"github.com/milvus-io/milvus/pkg/v3/proto/planpb"
)

// Chooses a native comparison from the owned operator and actual operand kinds.
func (v *exporter) exportComparison(graph *expressions.CallExpression, op planpb.OpType) (*planpb.Expr, error) {
	left, right := graph.ChildAt(0), graph.ChildAt(1)
	if !left.Type().EqualsToIgnoringNullability(right.Type()) || comparisonDomain(left.Type(), op) != noLimitation {
		return nil, unsupportedExport("comparison requires matching scalar domains and a supported operator")
	}
	if _, ok := left.(*expressions.LiteralExpression); ok {
		left, right = right, left
		op = reverseComparison(op)
	}
	if _, ok := left.(*expressions.ReferenceExpression); !ok {
		return nil, unsupportedExport("comparison requires a field operand")
	}
	column, err := exportField(v.imported.fields, left)
	if err != nil {
		return nil, err
	}
	switch right.(type) {
	case *expressions.ReferenceExpression:
		other, err := exportField(v.imported.fields, right)
		if err != nil {
			return nil, err
		}
		return &planpb.Expr{Expr: &planpb.Expr_CompareExpr{CompareExpr: &planpb.CompareExpr{LeftColumnInfo: column, RightColumnInfo: other, Op: op}}}, nil
	case *expressions.LiteralExpression:
		value, err := exportScalar(right)
		if err != nil {
			return nil, err
		}
		return &planpb.Expr{Expr: &planpb.Expr_UnaryRangeExpr{UnaryRangeExpr: &planpb.UnaryRangeExpr{ColumnInfo: column, Op: op, Value: value}}}, nil
	default:
		return nil, unsupportedExport("comparison operand has no native scalar representation")
	}
}

// Preserves comparison meaning when placing a right-hand reference first.
func reverseComparison(op planpb.OpType) planpb.OpType {
	switch op {
	case planpb.OpType_LessThan:
		return planpb.OpType_GreaterThan
	case planpb.OpType_LessEqual:
		return planpb.OpType_GreaterEqual
	case planpb.OpType_GreaterThan:
		return planpb.OpType_LessThan
	case planpb.OpType_GreaterEqual:
		return planpb.OpType_LessEqual
	default:
		return op
	}
}

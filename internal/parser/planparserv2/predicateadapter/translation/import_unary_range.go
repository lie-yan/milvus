package translation

import (
	"github.com/lie-yan/predicate-optimization/expressions"

	"github.com/milvus-io/milvus/pkg/v3/proto/planpb"
)

// Temporary source operands used to construct a comparison.
type unaryRangeOperands struct {
	// Validated field metadata and full Type.
	field fieldOperand
	// Exact admitted literal arm, payload and full Type.
	literal literalOperand
}

// Admits only the eight existing Bool/Int64 comparisons, without allocating graph or registry state.
func (v *importer) readUnaryRangeOperands(node *planpb.Expr_UnaryRangeExpr) (*unaryRangeOperands, error) {
	if node == nil || node.UnaryRangeExpr == nil {
		return nil, sourceDefect("missing unary comparison")
	}
	n := node.UnaryRangeExpr
	if n.ColumnInfo == nil || n.Value == nil || n.Value.Val == nil {
		return nil, sourceDefect("missing unary comparison operand")
	}
	if len(n.ProtoReflect().GetUnknown()) != 0 {
		return nil, unsupportedSource(excludedScopeOrVersion)
	}
	if n.Op == planpb.OpType_Invalid {
		return nil, sourceDefect("invalid unary comparison operator")
	}
	if len(n.ExtraValues) != 0 {
		return nil, unsupportedSource(unsupportedForm)
	}
	switch n.Op {
	case planpb.OpType_Equal, planpb.OpType_NotEqual, planpb.OpType_LessThan, planpb.OpType_LessEqual, planpb.OpType_GreaterThan, planpb.OpType_GreaterEqual:
	default:
		return nil, unsupportedSource(unsupportedForm)
	}
	field, reason, err := v.result.owner.readFieldOperand(n.ColumnInfo)
	if err != nil {
		return nil, err
	}
	if reason != noLimitation {
		return nil, unsupportedSource(reason)
	}
	boolean := v.result.owner.primitives.BoolNonNull.EqualsTo(field.typ)
	if !boolean && !v.result.owner.primitives.Int64NonNull.EqualsTo(field.typ) {
		return nil, unsupportedSource(excludedDomain)
	}
	if boolean && n.Op != planpb.OpType_Equal && n.Op != planpb.OpType_NotEqual {
		return nil, unsupportedSource(unsupportedForm)
	}
	literal, reason, err := v.result.owner.readLiteralOperand(n.Value, field.typ)
	if err != nil {
		return nil, err
	}
	if reason != noLimitation {
		return nil, unsupportedSource(reason)
	}
	return &unaryRangeOperands{field: field, literal: literal}, nil
}

// Constructs a comparison from field-first call arguments.
func (v *importer) importUnaryRange(node *planpb.Expr_UnaryRangeExpr) (expressions.Expression, error) {
	operands, err := v.readUnaryRangeOperands(node)
	if err != nil {
		return nil, err
	}
	owner := v.result.owner
	if err := v.ensureFields(); err != nil {
		return nil, err
	}
	field, err := importField(v.result.fields, operands.field)
	if err != nil {
		return nil, err
	}
	literal, err := owner.importLiteral(operands.literal)
	if err != nil {
		return nil, err
	}
	handle, err := owner.runtime.ComparisonFunction(node.UnaryRangeExpr.Op, field.Type(), literal.Type())
	if err != nil {
		return nil, newFailure(TranslationDefect, StageImport, err)
	}
	graph, err := expressions.NewCallExpression(owner.context, handle, []expressions.Expression{field, literal})
	if err != nil {
		return nil, newFailure(TranslationDefect, StageImport, err)
	}
	return graph, nil
}

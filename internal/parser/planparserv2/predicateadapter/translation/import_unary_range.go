package translation

import (
	"github.com/lie-yan/predicate-optimization/expressions"

	"github.com/milvus-io/milvus/internal/parser/planparserv2/predicateadapter/internal/assertions"
	"github.com/milvus-io/milvus/pkg/v3/proto/planpb"
)

// Temporary source operands used to construct a comparison.
type unaryRangeOperands struct {
	// Validated field metadata and full Type.
	field fieldOperand
	// Exact admitted literal arm, payload and full Type.
	literal literalOperand
}

// Admits scalar comparisons supported by the catalog without implicit conversion.
func (v *importer) readUnaryRangeOperands(node *planpb.Expr_UnaryRangeExpr) (unaryRangeOperands, error) {
	// Milvus supplies a populated plan; only translation coverage needs checking here.
	assertions.NotNilPointer(node)
	n := node.UnaryRangeExpr
	assertions.NotNilPointer(n)
	switch n.Op {
	case planpb.OpType_Equal, planpb.OpType_NotEqual, planpb.OpType_LessThan, planpb.OpType_LessEqual, planpb.OpType_GreaterThan, planpb.OpType_GreaterEqual:
	default:
		return unaryRangeOperands{}, unsupportedSource(unsupportedForm)
	}
	field, reason, err := v.result.owner.readFieldOperand(n.ColumnInfo)
	if err != nil {
		return unaryRangeOperands{}, err
	}
	if reason != noLimitation {
		return unaryRangeOperands{}, unsupportedSource(reason)
	}
	if reason := comparisonDomain(field.typ, n.Op); reason != noLimitation {
		return unaryRangeOperands{}, unsupportedSource(reason)
	}
	literal, reason, err := v.result.owner.readLiteralOperand(n.Value, field.typ)
	if err != nil {
		return unaryRangeOperands{}, err
	}
	if reason != noLimitation {
		return unaryRangeOperands{}, unsupportedSource(reason)
	}
	return unaryRangeOperands{field: field, literal: literal}, nil
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

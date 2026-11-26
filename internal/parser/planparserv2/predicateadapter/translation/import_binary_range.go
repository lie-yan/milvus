package translation

import (
	"github.com/lie-yan/predicate-optimization/expressions"
	"github.com/lie-yan/predicate-optimization/foundation"
	"github.com/lie-yan/predicate-optimization/types"

	"github.com/milvus-io/milvus/pkg/v3/proto/planpb"
)

// Source-backed operands for the generated lower comparison followed by the upper comparison.
type binaryRangeOperands struct {
	// Validated field metadata shared by both bounds.
	field fieldOperand
	// Exact lower-bound payload and Type.
	lower literalOperand
	// Exact upper-bound payload and Type.
	upper literalOperand
}

// Maps bound position and inclusivity to field-first comparison operators.
func rangeOperator(lower, inclusive bool) planpb.OpType {
	if lower {
		if inclusive {
			return planpb.OpType_GreaterEqual
		}
		return planpb.OpType_GreaterThan
	}
	if inclusive {
		return planpb.OpType_LessEqual
	}
	return planpb.OpType_LessThan
}

// Admits concrete numeric bounds only when both have an exact same-Type representation.
func (v *importer) readBinaryRangeOperands(node *planpb.Expr_BinaryRangeExpr) (*binaryRangeOperands, error) {
	if node == nil || node.BinaryRangeExpr == nil {
		return nil, sourceDefect("missing binary range")
	}
	n := node.BinaryRangeExpr
	if n.ColumnInfo == nil || n.LowerValue == nil || n.UpperValue == nil || n.LowerValue.Val == nil || n.UpperValue.Val == nil {
		return nil, sourceDefect("missing binary range operand")
	}
	if len(n.ProtoReflect().GetUnknown()) != 0 {
		return nil, unsupportedSource(excludedScopeOrVersion)
	}
	field, reason, err := v.result.owner.readFieldOperand(n.ColumnInfo)
	if err != nil {
		return nil, err
	}
	if reason != noLimitation {
		return nil, unsupportedSource(reason)
	}
	switch field.typ.(type) {
	case *types.IntType, *types.FloatType:
	default:
		return nil, unsupportedSource(excludedDomain)
	}
	lower, lr, err := v.result.owner.readRangeBound(n.LowerValue, field.typ)
	if err != nil {
		return nil, err
	}
	upper, ur, err := v.result.owner.readRangeBound(n.UpperValue, field.typ)
	if err != nil {
		return nil, err
	}
	if lr != noLimitation {
		return nil, unsupportedSource(lr)
	}
	if ur != noLimitation {
		return nil, unsupportedSource(ur)
	}
	// The parser rejects reversed bounds and equal bounds with an exclusive endpoint.
	var reversed, equal bool
	switch field.typ.(type) {
	case *types.IntType:
		l, u := n.LowerValue.GetInt64Val(), n.UpperValue.GetInt64Val()
		reversed, equal = l > u, l == u
	case *types.FloatType:
		l, u := n.LowerValue.GetFloatVal(), n.UpperValue.GetFloatVal()
		reversed, equal = l > u, l == u
	}
	if reversed || (equal && (!n.LowerInclusive || !n.UpperInclusive)) {
		return nil, sourceDefect("invalid binary range bound ordering")
	}
	return &binaryRangeOperands{field: field, lower: lower, upper: upper}, nil
}

// Constructs two ordered bound comparisons over one reference.
func (v *importer) importBinaryRange(node *planpb.Expr_BinaryRangeExpr) (expressions.Expression, error) {
	operands, err := v.readBinaryRangeOperands(node)
	if err != nil {
		return nil, err
	}
	if err := v.ensureFields(); err != nil {
		return nil, err
	}
	owner := v.result.owner
	field, err := importField(v.result.fields, operands.field)
	if err != nil {
		return nil, err
	}
	n := node.BinaryRangeExpr
	calls := make([]expressions.Expression, 2)
	for i, bound := range []literalOperand{operands.lower, operands.upper} {
		if err := v.ctx.Err(); err != nil {
			return nil, err
		}
		literal, err := owner.importLiteral(bound)
		if err != nil {
			return nil, err
		}
		inclusive := n.LowerInclusive
		if i == 1 {
			inclusive = n.UpperInclusive
		}
		handle, err := owner.runtime.ComparisonFunction(rangeOperator(i == 0, inclusive), field.Type(), literal.Type())
		if err != nil {
			return nil, newFailure(TranslationDefect, StageImport, err)
		}
		calls[i], err = expressions.NewCallExpression(owner.context, handle, []expressions.Expression{field, literal})
		if err != nil {
			return nil, newFailure(TranslationDefect, StageImport, err)
		}
	}
	graph, err := expressions.NewLogicalExpression(foundation.And, calls)
	if err != nil {
		return nil, newFailure(TranslationDefect, StageImport, err)
	}
	return graph, nil
}

// Keeps integer-carried floating bounds outside the no-conversion path.
func (t *Translator) readRangeBound(value *planpb.GenericValue, typ types.Type) (literalOperand, limitation, error) {
	if _, floating := typ.(*types.FloatType); floating {
		if integer, ok := value.Val.(*planpb.GenericValue_Int64Val); ok && integer != nil {
			return literalOperand{}, excludedDomain, nil
		}
	}
	return t.readLiteralOperand(value, typ)
}

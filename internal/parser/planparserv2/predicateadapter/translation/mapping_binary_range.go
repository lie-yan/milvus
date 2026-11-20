package translation

import (
	"github.com/lie-yan/predicate-optimization/expressions"
	"github.com/lie-yan/predicate-optimization/foundation"
	"github.com/lie-yan/predicate-optimization/types"

	"github.com/milvus-io/milvus/pkg/v3/proto/planpb"
)

// Source-backed operands for the generated lower comparison followed by the upper comparison.
type binaryRangeOperands struct {
	// Shared field identity and occurrence metadata.
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
func (a *admitter) admitBinaryRange(source *planpb.Expr, node *planpb.Expr_BinaryRangeExpr) (admissionResult, error) {
	if node == nil || node.BinaryRangeExpr == nil {
		return admissionResult{}, sourceDefect("missing binary range")
	}
	n := node.BinaryRangeExpr
	if n.ColumnInfo == nil || n.LowerValue == nil || n.UpperValue == nil || n.LowerValue.Val == nil || n.UpperValue.Val == nil {
		return admissionResult{}, sourceDefect("missing binary range operand")
	}
	if len(n.ProtoReflect().GetUnknown()) != 0 {
		return admissionResult{limitation: excludedScopeOrVersion}, nil
	}
	field, reason, err := a.translator.admitField(n.ColumnInfo)
	if err != nil {
		return admissionResult{}, err
	}
	if reason != noLimitation {
		return admissionResult{limitation: reason}, nil
	}
	switch field.typ.(type) {
	case *types.IntType, *types.FloatType:
	default:
		return admissionResult{limitation: excludedDomain}, nil
	}
	lower, lr, err := a.translator.admitRangeBound(n.LowerValue, field.typ)
	if err != nil {
		return admissionResult{}, err
	}
	upper, ur, err := a.translator.admitRangeBound(n.UpperValue, field.typ)
	if err != nil {
		return admissionResult{}, err
	}
	if lr != noLimitation {
		return admissionResult{limitation: lr}, nil
	}
	if ur != noLimitation {
		return admissionResult{limitation: ur}, nil
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
		return admissionResult{}, sourceDefect("invalid binary range bound ordering")
	}
	id := routeID(len(a.routes))
	a.routes = append(a.routes, admittedRoute{source: source, kind: binaryRangeRoute, typ: a.translator.primitives.BoolNonNull, binaryRange: &binaryRangeOperands{field: field, lower: lower, upper: upper}})
	return admissionResult{route: id}, nil
}

// Constructs two ordered calls over one reference without inventing source occurrences for generated nodes.
func (v *importer) importBinaryRange(route admittedRoute) (occurrenceID, error) {
	if err := v.ensureFields(); err != nil {
		return 0, err
	}
	owner := v.result.admitted.owner
	field, err := importField(v.result.fields, route.binaryRange.field)
	if err != nil {
		return 0, err
	}
	source := route.source.GetBinaryRangeExpr()
	calls := make([]expressions.Expression, 2)
	for i, bound := range []literalOperand{route.binaryRange.lower, route.binaryRange.upper} {
		if err := v.ctx.Err(); err != nil {
			return 0, err
		}
		literal, err := owner.importLiteral(bound)
		if err != nil {
			return 0, err
		}
		inclusive := source.LowerInclusive
		if i == 1 {
			inclusive = source.UpperInclusive
		}
		handle, err := owner.runtime.ComparisonFunction(rangeOperator(i == 0, inclusive), field.Type(), literal.Type())
		if err != nil {
			return 0, newFailure(TranslationDefect, StageImport, err)
		}
		calls[i], err = expressions.NewCallExpression(owner.context, handle, []expressions.Expression{field, literal})
		if err != nil {
			return 0, newFailure(TranslationDefect, StageImport, err)
		}
	}
	graph, err := expressions.NewLogicalExpression(foundation.And, calls)
	if err != nil {
		return 0, newFailure(TranslationDefect, StageImport, err)
	}
	id := appendOccurrence(&v.result.occurrences, route.source, graph, nil)
	v.result.occurrences[id].binaryRange = route.binaryRange
	return id, nil
}

// Checks generated structure and each bound's operator, field and literal against its retained role.
func (v *graphChecker) checkBinaryRange(id occurrenceID, graph expressions.Expression) error {
	record := v.imported.occurrences[id]
	node, ok := graph.(*expressions.LogicalExpression)
	if !ok || node.Operator() != foundation.And || node.ChildCount() != 2 || !v.imported.admitted.owner.primitives.BoolNonNull.EqualsTo(node.Type()) {
		return translationDefect(v.stage, "binary range changed conjunction")
	}
	source := record.source.GetBinaryRangeExpr()
	for i, bound := range []literalOperand{record.binaryRange.lower, record.binaryRange.upper} {
		if err := v.ctx.Err(); err != nil {
			return err
		}
		call, ok := node.ChildAt(i).(*expressions.CallExpression)
		if !ok {
			return translationDefect(v.stage, "binary range changed bound form")
		}
		inclusive := source.LowerInclusive
		if i == 1 {
			inclusive = source.UpperInclusive
		}
		if err := v.checkComparisonCall(record.imported.ChildAt(i), call, rangeOperator(i == 0, inclusive)); err != nil {
			return err
		}
		if err := checkField(v.imported.fields, record.binaryRange.field, call.ChildAt(0), v.stage); err != nil {
			return err
		}
		if err := checkLiteral(bound, call.ChildAt(1), v.stage); err != nil {
			return err
		}
	}
	return nil
}

// Emits the specialized range only for a retained range occurrence, reading actual checked graph payloads.
func (v *exporter) exportBinaryRange(id occurrenceID, graph expressions.Expression) (*planpb.Expr, error) {
	record := v.imported.occurrences[id]
	source := record.source.GetBinaryRangeExpr()
	field, err := exportField(v.imported.fields, record.binaryRange.field, graph.ChildAt(0).ChildAt(0))
	if err != nil {
		return nil, err
	}
	bounds := make([]*planpb.GenericValue, 2)
	inclusive := [2]bool{}
	for i, bound := range []literalOperand{record.binaryRange.lower, record.binaryRange.upper} {
		if err := v.ctx.Err(); err != nil {
			return nil, err
		}
		call := graph.ChildAt(i).(*expressions.CallExpression)
		bounds[i], err = exportLiteral(bound, call.ChildAt(1))
		if err != nil {
			return nil, err
		}
		op, err := v.imported.admitted.owner.runtime.ComparisonOperator(call.Handle())
		if err != nil {
			return nil, newFailure(TranslationDefect, StageExport, err)
		}
		inclusive[i] = op == planpb.OpType_GreaterEqual || op == planpb.OpType_LessEqual
	}
	return &planpb.Expr{IsTemplate: record.source.IsTemplate, Expr: &planpb.Expr_BinaryRangeExpr{BinaryRangeExpr: &planpb.BinaryRangeExpr{
		ColumnInfo: field, LowerValue: bounds[0], UpperValue: bounds[1], LowerInclusive: inclusive[0], UpperInclusive: inclusive[1],
		LowerTemplateVariableName: source.LowerTemplateVariableName, UpperTemplateVariableName: source.UpperTemplateVariableName,
	}}}, nil
}

// Independently verifies range metadata, both bounds and inclusivity without reconstructing expected output.
func (v *candidateChecker) checkBinaryRange(record occurrence, candidate *planpb.Expr) error {
	if candidate == nil || len(candidate.ProtoReflect().GetUnknown()) != 0 || candidate.IsTemplate != record.source.IsTemplate {
		return translationDefect(StageCandidateValidation, "binary range changed envelope")
	}
	node, ok := candidate.Expr.(*planpb.Expr_BinaryRangeExpr)
	if !ok || node == nil || node.BinaryRangeExpr == nil {
		return translationDefect(StageCandidateValidation, "binary range changed form")
	}
	n := node.BinaryRangeExpr
	source := record.source.GetBinaryRangeExpr()
	if len(n.ProtoReflect().GetUnknown()) != 0 || n.LowerInclusive != source.LowerInclusive || n.UpperInclusive != source.UpperInclusive || n.LowerTemplateVariableName != source.LowerTemplateVariableName || n.UpperTemplateVariableName != source.UpperTemplateVariableName {
		return translationDefect(StageCandidateValidation, "binary range changed bound metadata")
	}
	if err := checkFieldCandidate(record.binaryRange.field, n.ColumnInfo); err != nil {
		return err
	}
	if err := checkLiteralCandidate(record.binaryRange.lower, n.LowerValue); err != nil {
		return err
	}
	return checkLiteralCandidate(record.binaryRange.upper, n.UpperValue)
}

// Keeps integer-carried floating bounds outside the no-conversion path.
func (t *Translator) admitRangeBound(value *planpb.GenericValue, typ types.Type) (literalOperand, limitation, error) {
	if _, floating := typ.(*types.FloatType); floating {
		if integer, ok := value.Val.(*planpb.GenericValue_Int64Val); ok && integer != nil {
			return literalOperand{}, excludedDomain, nil
		}
	}
	return t.admitLiteral(value, typ)
}

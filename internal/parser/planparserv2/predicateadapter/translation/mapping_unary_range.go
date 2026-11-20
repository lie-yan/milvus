package translation

import (
	"github.com/lie-yan/predicate-optimization/expressions"

	"github.com/milvus-io/milvus/pkg/v3/proto/planpb"
)

// Embedded operand associations shared by the admitted route and imported occurrence.
type unaryRangeOperands struct {
	// Expected field identity, full Type and occurrence metadata.
	field fieldOperand
	// Exact admitted literal arm, payload and full Type.
	literal literalOperand
}

// Admits only the eight existing Bool/Int64 comparisons, without allocating graph or registry state.
func (a *admitter) admitUnaryRange(source *planpb.Expr, node *planpb.Expr_UnaryRangeExpr) (admissionResult, error) {
	if node == nil || node.UnaryRangeExpr == nil {
		return admissionResult{}, sourceDefect("missing unary comparison")
	}
	n := node.UnaryRangeExpr
	if n.ColumnInfo == nil || n.Value == nil || n.Value.Val == nil {
		return admissionResult{}, sourceDefect("missing unary comparison operand")
	}
	if len(n.ProtoReflect().GetUnknown()) != 0 {
		return admissionResult{limitation: excludedScopeOrVersion}, nil
	}
	if n.Op == planpb.OpType_Invalid {
		return admissionResult{}, sourceDefect("invalid unary comparison operator")
	}
	if len(n.ExtraValues) != 0 {
		return admissionResult{limitation: unsupportedForm}, nil
	}
	switch n.Op {
	case planpb.OpType_Equal, planpb.OpType_NotEqual, planpb.OpType_LessThan, planpb.OpType_LessEqual, planpb.OpType_GreaterThan, planpb.OpType_GreaterEqual:
	default:
		return admissionResult{limitation: unsupportedForm}, nil
	}
	field, reason, err := a.translator.admitField(n.ColumnInfo)
	if err != nil {
		return admissionResult{}, err
	}
	if reason != noLimitation {
		return admissionResult{limitation: reason}, nil
	}
	boolean := a.translator.primitives.BoolNonNull.EqualsTo(field.typ)
	if !boolean && !a.translator.primitives.Int64NonNull.EqualsTo(field.typ) {
		return admissionResult{limitation: excludedDomain}, nil
	}
	if boolean && n.Op != planpb.OpType_Equal && n.Op != planpb.OpType_NotEqual {
		return admissionResult{limitation: unsupportedForm}, nil
	}
	literal, reason, err := a.translator.admitLiteral(n.Value, field.typ)
	if err != nil {
		return admissionResult{}, err
	}
	if reason != noLimitation {
		return admissionResult{limitation: reason}, nil
	}
	id := routeID(len(a.routes))
	a.routes = append(a.routes, admittedRoute{source: source, kind: unaryRangeRoute, typ: a.translator.primitives.BoolNonNull, unary: &unaryRangeOperands{field: field, literal: literal}})
	return admissionResult{route: id}, nil
}

// Creates the owning field registry only after whole-source admission and binds field-first call arguments.
func (v *importer) importUnaryRange(route admittedRoute) (occurrenceID, error) {
	owner := v.result.admitted.owner
	if err := v.ensureFields(); err != nil {
		return 0, err
	}
	field, err := importField(v.result.fields, route.unary.field)
	if err != nil {
		return 0, err
	}
	literal, err := owner.importLiteral(route.unary.literal)
	if err != nil {
		return 0, err
	}
	handle, err := owner.runtime.ComparisonFunction(route.source.GetUnaryRangeExpr().Op, field.Type(), literal.Type())
	if err != nil {
		return 0, newFailure(TranslationDefect, StageImport, err)
	}
	graph, err := expressions.NewCallExpression(owner.context, handle, []expressions.Expression{field, literal})
	if err != nil {
		return 0, newFailure(TranslationDefect, StageImport, err)
	}
	id := appendOccurrence(&v.result.occurrences, route.source, graph, nil)
	v.result.occurrences[id].unary = route.unary
	return id, nil
}

// Checks retained provider, canonical handle/operator and both ordered operand roles on the actual call.
func (v *graphChecker) checkUnaryRange(id occurrenceID, graph *expressions.CallExpression) error {
	if id < 0 || int(id) >= len(v.imported.occurrences) {
		return translationDefect(v.stage, "missing comparison occurrence")
	}
	record := v.imported.occurrences[id]
	if record.unary == nil {
		return translationDefect(v.stage, "call has no unary comparison route")
	}
	if err := v.checkComparisonCall(record.imported, graph, record.source.GetUnaryRangeExpr().Op); err != nil {
		return err
	}
	if err := checkField(v.imported.fields, record.unary.field, graph.ChildAt(0), v.stage); err != nil {
		return err
	}
	return checkLiteral(record.unary.literal, graph.ChildAt(1), v.stage)
}

// Emits the specialized Milvus node from actual graph operands/operator and retained template metadata.
func (v *exporter) exportUnaryRange(id occurrenceID, graph *expressions.CallExpression) (*planpb.Expr, error) {
	record := v.imported.occurrences[id]
	field, err := exportField(v.imported.fields, record.unary.field, graph.ChildAt(0))
	if err != nil {
		return nil, err
	}
	literal, err := exportLiteral(record.unary.literal, graph.ChildAt(1))
	if err != nil {
		return nil, err
	}
	op, err := v.imported.admitted.owner.runtime.ComparisonOperator(graph.Handle())
	if err != nil {
		return nil, newFailure(TranslationDefect, StageExport, err)
	}
	return &planpb.Expr{IsTemplate: record.source.IsTemplate, Expr: &planpb.Expr_UnaryRangeExpr{UnaryRangeExpr: &planpb.UnaryRangeExpr{
		ColumnInfo: field, Op: op, Value: literal, TemplateVariableName: record.source.GetUnaryRangeExpr().TemplateVariableName,
	}}}, nil
}

// Independently checks emitted comparison fields, template metadata and exact operand content.
func (v *candidateChecker) checkUnaryRange(record occurrence, candidate *planpb.Expr) error {
	if candidate == nil || len(candidate.ProtoReflect().GetUnknown()) != 0 || candidate.IsTemplate != record.source.IsTemplate {
		return translationDefect(StageCandidateValidation, "comparison candidate changed envelope")
	}
	node, ok := candidate.Expr.(*planpb.Expr_UnaryRangeExpr)
	if !ok || node == nil || node.UnaryRangeExpr == nil || record.unary == nil {
		return translationDefect(StageCandidateValidation, "comparison candidate changed form")
	}
	n := node.UnaryRangeExpr
	source := record.source.GetUnaryRangeExpr()
	if len(n.ProtoReflect().GetUnknown()) != 0 || len(n.ExtraValues) != 0 || n.Op != source.Op || n.TemplateVariableName != source.TemplateVariableName {
		return translationDefect(StageCandidateValidation, "comparison candidate changed operator or metadata")
	}
	if err := checkFieldCandidate(record.unary.field, n.ColumnInfo); err != nil {
		return err
	}
	return checkLiteralCandidate(record.unary.literal, n.Value)
}

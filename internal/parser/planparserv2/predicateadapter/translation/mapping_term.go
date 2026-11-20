package translation

import (
	"github.com/lie-yan/predicate-optimization/expressions"
	"github.com/lie-yan/predicate-optimization/foundation"
	"github.com/lie-yan/predicate-optimization/types"

	"github.com/milvus-io/milvus/pkg/v3/proto/planpb"
)

// Source-backed target and ordered candidates, independent of logical NOT wrappers.
type termOperands struct {
	// Target identity, full Type and occurrence metadata.
	field fieldOperand
	// Exact candidate payloads in source order.
	candidates []literalOperand
}

// Admits homogeneous nonempty scalar membership while preserving order and duplicates.
func (a *admitter) admitTerm(source *planpb.Expr, node *planpb.Expr_TermExpr) (admissionResult, error) {
	if node == nil || node.TermExpr == nil {
		return admissionResult{}, sourceDefect("missing term expression")
	}
	n := node.TermExpr
	if n.ColumnInfo == nil {
		return admissionResult{}, sourceDefect("missing term target")
	}
	if len(n.ProtoReflect().GetUnknown()) != 0 {
		return admissionResult{limitation: excludedScopeOrVersion}, nil
	}
	if n.IsInField || len(n.Values) == 0 {
		return admissionResult{limitation: unsupportedForm}, nil
	}
	field, reason, err := a.translator.admitField(n.ColumnInfo)
	if err != nil {
		return admissionResult{}, err
	}
	if reason != noLimitation {
		return admissionResult{limitation: reason}, nil
	}
	candidates := make([]literalOperand, len(n.Values))
	for i, value := range n.Values {
		if err := a.ctx.Err(); err != nil {
			return admissionResult{}, err
		}
		candidate, r, err := a.translator.admitTermCandidate(value, field.typ)
		if err != nil {
			return admissionResult{}, err
		}
		if r != noLimitation {
			reason = r
		}
		candidates[i] = candidate
	}
	if reason != noLimitation {
		return admissionResult{limitation: reason}, nil
	}
	id := routeID(len(a.routes))
	a.routes = append(a.routes, admittedRoute{source: source, kind: termRoute, typ: a.translator.primitives.BoolNonNull, term: &termOperands{field: field, candidates: candidates}})
	return admissionResult{route: id}, nil
}

// Classifies heterogeneous/coercion-dependent candidates before invoking exact scalar codecs.
func (t *Translator) admitTermCandidate(value *planpb.GenericValue, typ types.Type) (literalOperand, limitation, error) {
	if value == nil || value.Val == nil {
		return literalOperand{}, noLimitation, sourceDefect("missing term candidate")
	}
	if len(value.ProtoReflect().GetUnknown()) != 0 {
		return literalOperand{}, excludedScopeOrVersion, nil
	}
	var matches bool
	switch v := value.Val.(type) {
	case *planpb.GenericValue_BoolVal:
		if v == nil {
			return literalOperand{}, noLimitation, sourceDefect("missing Bool candidate arm")
		}
		_, matches = typ.(*types.BoolType)
	case *planpb.GenericValue_Int64Val:
		if v == nil {
			return literalOperand{}, noLimitation, sourceDefect("missing integer candidate arm")
		}
		_, matches = typ.(*types.IntType)
	case *planpb.GenericValue_FloatVal:
		if v == nil {
			return literalOperand{}, noLimitation, sourceDefect("missing floating candidate arm")
		}
		_, matches = typ.(*types.FloatType)
	case *planpb.GenericValue_StringVal:
		if v == nil {
			return literalOperand{}, noLimitation, sourceDefect("missing string candidate arm")
		}
		_, matches = typ.(*types.VarCharType)
	}
	if !matches {
		return literalOperand{}, excludedDomain, nil
	}
	return t.admitLiteral(value, typ)
}

// Builds a dedicated IN node with the target first and literal candidates in their admitted order.
func (v *importer) importTerm(route admittedRoute) (occurrenceID, error) {
	if err := v.ensureFields(); err != nil {
		return 0, err
	}
	field, err := importField(v.result.fields, route.term.field)
	if err != nil {
		return 0, err
	}
	owner := v.result.admitted.owner
	args := make([]expressions.Expression, len(route.term.candidates)+1)
	args[0] = field
	for i, candidate := range route.term.candidates {
		if err := v.ctx.Err(); err != nil {
			return 0, err
		}
		args[i+1], err = owner.importLiteral(candidate)
		if err != nil {
			return 0, err
		}
	}
	graph, err := expressions.NewSpecialFormExpression(foundation.SpecialFormIn, owner.primitives.BoolNonNull, args)
	if err != nil {
		return 0, newFailure(TranslationDefect, StageImport, err)
	}
	id := appendOccurrence(&v.result.occurrences, route.source, graph, nil)
	v.result.occurrences[id].term = route.term
	return id, nil
}

// Checks the actual form, target, full Types and each candidate position against retained roles.
func (v *graphChecker) checkTerm(id occurrenceID, graph expressions.Expression) error {
	record := v.imported.occurrences[id]
	node, ok := graph.(*expressions.SpecialFormExpression)
	if !ok || node.Form() != foundation.SpecialFormIn || node.ChildCount() != len(record.term.candidates)+1 || !v.imported.admitted.owner.primitives.BoolNonNull.EqualsTo(node.Type()) {
		return translationDefect(v.stage, "term changed form, arity or result Type")
	}
	if err := checkField(v.imported.fields, record.term.field, node.ChildAt(0), v.stage); err != nil {
		return err
	}
	for i, candidate := range record.term.candidates {
		if err := v.ctx.Err(); err != nil {
			return err
		}
		if err := checkLiteral(candidate, node.ChildAt(i+1), v.stage); err != nil {
			return err
		}
	}
	return nil
}

// Emits fresh target and candidate messages from the checked graph, retaining source template metadata.
func (v *exporter) exportTerm(id occurrenceID, graph expressions.Expression) (*planpb.Expr, error) {
	record := v.imported.occurrences[id]
	field, err := exportField(v.imported.fields, record.term.field, graph.ChildAt(0))
	if err != nil {
		return nil, err
	}
	values := make([]*planpb.GenericValue, len(record.term.candidates))
	for i, candidate := range record.term.candidates {
		if err := v.ctx.Err(); err != nil {
			return nil, err
		}
		values[i], err = exportLiteral(candidate, graph.ChildAt(i+1))
		if err != nil {
			return nil, err
		}
	}
	return &planpb.Expr{IsTemplate: record.source.IsTemplate, Expr: &planpb.Expr_TermExpr{TermExpr: &planpb.TermExpr{
		ColumnInfo: field, Values: values, TemplateVariableName: record.source.GetTermExpr().TemplateVariableName,
	}}}, nil
}

// Independently verifies emitted membership shape, metadata and ordered candidate payloads.
func (v *candidateChecker) checkTerm(record occurrence, candidate *planpb.Expr) error {
	if candidate == nil || len(candidate.ProtoReflect().GetUnknown()) != 0 || candidate.IsTemplate != record.source.IsTemplate {
		return translationDefect(StageCandidateValidation, "term changed envelope")
	}
	node, ok := candidate.Expr.(*planpb.Expr_TermExpr)
	if !ok || node == nil || node.TermExpr == nil {
		return translationDefect(StageCandidateValidation, "term changed form")
	}
	n := node.TermExpr
	if len(n.ProtoReflect().GetUnknown()) != 0 || n.IsInField || len(n.Values) != len(record.term.candidates) || n.TemplateVariableName != record.source.GetTermExpr().TemplateVariableName {
		return translationDefect(StageCandidateValidation, "term changed membership metadata")
	}
	if err := checkFieldCandidate(record.term.field, n.ColumnInfo); err != nil {
		return err
	}
	for i, operand := range record.term.candidates {
		if err := v.ctx.Err(); err != nil {
			return err
		}
		if err := checkLiteralCandidate(operand, n.Values[i]); err != nil {
			return err
		}
	}
	return nil
}

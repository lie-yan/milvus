package translation

import (
	"github.com/lie-yan/predicate-optimization/expressions"
	"github.com/lie-yan/predicate-optimization/foundation"
	valueinterfaces "github.com/lie-yan/predicate-optimization/values/interfaces"

	"github.com/milvus-io/milvus/pkg/v3/proto/planpb"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
)

// Checks the local native shape without visiting children or selecting fallback.
func logicalShape(source *planpb.Expr) (routeKind, []*planpb.Expr, limitation, error) {
	if source == nil {
		return 0, nil, noLimitation, merr.WrapErrServiceInternalMsg("predicate translation: missing logical source")
	}
	if len(source.ProtoReflect().GetUnknown()) != 0 {
		return 0, nil, excludedScopeOrVersion, nil
	}
	switch node := source.Expr.(type) {
	case *planpb.Expr_AlwaysTrueExpr:
		if node == nil || node.AlwaysTrueExpr == nil {
			break
		}
		if len(node.AlwaysTrueExpr.ProtoReflect().GetUnknown()) != 0 {
			return 0, nil, excludedScopeOrVersion, nil
		}
		return trueRoute, nil, noLimitation, nil
	case *planpb.Expr_UnaryExpr:
		if node == nil || node.UnaryExpr == nil || node.UnaryExpr.Child == nil {
			break
		}
		n := node.UnaryExpr
		if len(n.ProtoReflect().GetUnknown()) != 0 {
			return 0, nil, excludedScopeOrVersion, nil
		}
		if n.Op == planpb.UnaryExpr_Invalid {
			break
		}
		if n.Op != planpb.UnaryExpr_Not {
			return 0, nil, unsupportedForm, nil
		}
		return notRoute, []*planpb.Expr{n.Child}, noLimitation, nil
	case *planpb.Expr_BinaryExpr:
		if node == nil || node.BinaryExpr == nil || node.BinaryExpr.Left == nil || node.BinaryExpr.Right == nil {
			break
		}
		n := node.BinaryExpr
		if len(n.ProtoReflect().GetUnknown()) != 0 {
			return 0, nil, excludedScopeOrVersion, nil
		}
		switch n.Op {
		case planpb.BinaryExpr_LogicalAnd:
			return andRoute, []*planpb.Expr{n.Left, n.Right}, noLimitation, nil
		case planpb.BinaryExpr_LogicalOr:
			return orRoute, []*planpb.Expr{n.Left, n.Right}, noLimitation, nil
		case planpb.BinaryExpr_Invalid:
			break
		default:
			return 0, nil, unsupportedForm, nil
		}
	}
	return 0, nil, noLimitation, merr.WrapErrServiceInternalMsg("predicate translation: malformed native Boolean expression")
}

// Requires every exposed child, preserving defects even when a sibling lacks coverage.
func (a *admitter) admitLogical(source *planpb.Expr) (admissionResult, error) {
	kind, sources, reason, err := logicalShape(source)
	if err != nil {
		return admissionResult{}, newFailure(UnsafeSource, StageAdmission, err)
	}
	if reason != noLimitation {
		return admissionResult{limitation: reason}, nil
	}
	start := len(a.routes)
	children := make([]routeID, 0, len(sources))
	for _, child := range sources {
		result, err := a.admitExpr(child)
		if err != nil {
			a.routes = a.routes[:start]
			return admissionResult{}, err
		}
		if result.limitation != noLimitation {
			if reason == noLimitation {
				reason = result.limitation
			}
			continue
		}
		if !a.translator.primitives.BoolNonNull.EqualsTo(a.routes[result.route].typ) {
			if reason == noLimitation {
				reason = excludedDomain
			}
			continue
		}
		children = append(children, result.route)
	}
	if reason != noLimitation {
		a.routes = a.routes[:start]
		return admissionResult{limitation: reason}, nil
	}
	id := routeID(len(a.routes))
	a.routes = append(a.routes, admittedRoute{source: source, kind: kind, typ: a.translator.primitives.BoolNonNull, children: children})
	return admissionResult{route: id}, nil
}

// Maps only admitted logical routes; TRUE is handled as a literal separately.
func logicalOperator(kind routeKind) foundation.LogicalOperator {
	switch kind {
	case notRoute:
		return foundation.Not
	case andRoute:
		return foundation.And
	case orRoute:
		return foundation.Or
	default:
		return 0
	}
}

// Constructs ordered native nodes without folding, flattening or evaluating children.
func (v *importer) importLogical(route admittedRoute) (occurrenceID, error) {
	children := make([]occurrenceID, 0, len(route.children))
	graphs := make([]expressions.Expression, 0, len(route.children))
	for _, child := range route.children {
		id, err := v.importExpr(child)
		if err != nil {
			return 0, err
		}
		children = append(children, id)
		graphs = append(graphs, v.result.occurrences[id].imported)
	}
	var graph expressions.Expression
	var err error
	if route.kind == trueRoute {
		value, valueErr := v.result.admitted.owner.factory.NewBool(v.result.admitted.owner.primitives.BoolNonNull, true)
		if valueErr != nil {
			return 0, newFailure(TranslationDefect, StageImport, valueErr)
		}
		graph, err = expressions.NewLiteralExpression(value)
	} else {
		graph, err = expressions.NewLogicalExpression(logicalOperator(route.kind), graphs)
	}
	if err != nil {
		return 0, newFailure(TranslationDefect, StageImport, err)
	}
	return appendOccurrence(&v.result.occurrences, route.source, graph, children), nil
}

// Requires the TRUE payload through the public Value interface and exact result Type.
func (t *Translator) isTrueLiteral(graph expressions.Expression) bool {
	literal, ok := graph.(*expressions.LiteralExpression)
	if !ok || literal == nil || !t.primitives.BoolNonNull.EqualsTo(literal.Type()) {
		return false
	}
	value, ok := literal.Value().(valueinterfaces.BoolValue)
	return ok && !value.IsNull() && value.Bool()
}

// Checks local native correspondence before descending through ordered occurrences.
func (v *graphChecker) checkLogical(id occurrenceID, graph expressions.Expression) error {
	if id < 0 || int(id) >= len(v.imported.occurrences) {
		return translationDefect(v.stage, "missing logical occurrence")
	}
	record := v.imported.occurrences[id]
	owner := v.imported.admitted.owner
	if !owner.primitives.BoolNonNull.EqualsTo(graph.Type()) {
		return translationDefect(v.stage, "logical result Type changed")
	}
	switch original := record.imported.(type) {
	case *expressions.LiteralExpression:
		if len(record.children) != 0 || !owner.isTrueLiteral(graph) || !owner.isTrueLiteral(original) {
			return translationDefect(v.stage, "TRUE literal changed")
		}
	case *expressions.LogicalExpression:
		node, ok := graph.(*expressions.LogicalExpression)
		if !ok || node.Operator() != original.Operator() || node.ChildCount() != len(record.children) || node.ChildCount() != original.ChildCount() {
			return translationDefect(v.stage, "logical operator or arity changed")
		}
		for i, child := range record.children {
			if err := v.checkExpr(child, node.ChildAt(i)); err != nil {
				return err
			}
		}
	default:
		return translationDefect(v.stage, "native graph replaced another route")
	}
	return nil
}

// Reconstructs native payloads from the checked actual graph, borrowing only envelope flags.
func (v *exporter) exportLogical(id occurrenceID, graph expressions.Expression) (*planpb.Expr, error) {
	record := v.imported.occurrences[id]
	candidate := &planpb.Expr{IsTemplate: record.source.IsTemplate}
	if _, ok := graph.(*expressions.LiteralExpression); ok {
		if !v.imported.admitted.owner.isTrueLiteral(graph) {
			return nil, translationDefect(StageExport, "expected TRUE literal")
		}
		candidate.Expr = &planpb.Expr_AlwaysTrueExpr{AlwaysTrueExpr: &planpb.AlwaysTrueExpr{}}
		return candidate, nil
	}
	node := graph.(*expressions.LogicalExpression)
	children := make([]*planpb.Expr, len(record.children))
	for i, child := range record.children {
		output, err := v.exportExpr(child, node.ChildAt(i))
		if err != nil {
			return nil, err
		}
		children[i] = output
	}
	switch node.Operator() {
	case foundation.Not:
		candidate.Expr = &planpb.Expr_UnaryExpr{UnaryExpr: &planpb.UnaryExpr{Op: planpb.UnaryExpr_Not, Child: children[0]}}
	case foundation.And, foundation.Or:
		op := planpb.BinaryExpr_LogicalAnd
		if node.Operator() == foundation.Or {
			op = planpb.BinaryExpr_LogicalOr
		}
		candidate.Expr = &planpb.Expr_BinaryExpr{BinaryExpr: &planpb.BinaryExpr{Op: op, Left: children[0], Right: children[1]}}
	default:
		return nil, translationDefect(StageExport, "unmapped logical operator")
	}
	return candidate, nil
}

// Inspects emitted native payloads and metadata independently, then checks every child.
func (v *candidateChecker) checkLogical(record occurrence, candidate *planpb.Expr) error {
	kind, children, reason, err := logicalShape(candidate)
	if err != nil {
		return newFailure(TranslationDefect, StageCandidateValidation, err)
	}
	if reason != noLimitation || candidate.IsTemplate != record.source.IsTemplate || len(children) != len(record.children) {
		return translationDefect(StageCandidateValidation, "native candidate shape or metadata changed")
	}
	switch original := record.imported.(type) {
	case *expressions.LiteralExpression:
		if kind != trueRoute {
			return translationDefect(StageCandidateValidation, "TRUE candidate changed form")
		}
	case *expressions.LogicalExpression:
		if logicalOperator(kind) != original.Operator() {
			return translationDefect(StageCandidateValidation, "candidate logical operator changed")
		}
	}
	for i, child := range children {
		if err := v.checkExpr(record.children[i], child); err != nil {
			return err
		}
	}
	return nil
}

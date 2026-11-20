package translation

import (
	"github.com/lie-yan/predicate-optimization/expressions"
	"github.com/lie-yan/predicate-optimization/types"

	"github.com/milvus-io/milvus/pkg/v3/proto/planpb"
)

// Ordered source-backed field roles, including each occurrence's full metadata.
type compareOperands struct {
	// Left reference identity and Type.
	left fieldOperand
	// Right reference identity and Type.
	right fieldOperand
}

// Admits same-Type numeric comparisons and Bool equality without casts or evaluation.
func (a *admitter) admitCompare(source *planpb.Expr, node *planpb.Expr_CompareExpr) (admissionResult, error) {
	if node == nil || node.CompareExpr == nil {
		return admissionResult{}, sourceDefect("missing field comparison")
	}
	n := node.CompareExpr
	if n.LeftColumnInfo == nil || n.RightColumnInfo == nil {
		return admissionResult{}, sourceDefect("missing comparison field")
	}
	if len(n.ProtoReflect().GetUnknown()) != 0 {
		return admissionResult{limitation: excludedScopeOrVersion}, nil
	}
	if n.Op == planpb.OpType_Invalid {
		return admissionResult{}, sourceDefect("invalid field comparison operator")
	}
	switch n.Op {
	case planpb.OpType_Equal, planpb.OpType_NotEqual, planpb.OpType_LessThan, planpb.OpType_LessEqual, planpb.OpType_GreaterThan, planpb.OpType_GreaterEqual:
	default:
		return admissionResult{limitation: unsupportedForm}, nil
	}
	left, leftReason, err := a.translator.admitField(n.LeftColumnInfo)
	if err != nil {
		return admissionResult{}, err
	}
	// Inspect both fields so an unsupported left domain cannot hide an invalid right field.
	right, rightReason, err := a.translator.admitField(n.RightColumnInfo)
	if err != nil {
		return admissionResult{}, err
	}
	if leftReason != noLimitation {
		return admissionResult{limitation: leftReason}, nil
	}
	if rightReason != noLimitation {
		return admissionResult{limitation: rightReason}, nil
	}
	if !left.typ.EqualsTo(right.typ) {
		return admissionResult{limitation: excludedDomain}, nil
	}
	switch left.typ.(type) {
	case *types.IntType, *types.FloatType:
	case *types.BoolType:
		if n.Op != planpb.OpType_Equal && n.Op != planpb.OpType_NotEqual {
			return admissionResult{limitation: unsupportedForm}, nil
		}
	default:
		return admissionResult{limitation: excludedDomain}, nil
	}
	id := routeID(len(a.routes))
	a.routes = append(a.routes, admittedRoute{source: source, kind: compareRoute, typ: a.translator.primitives.BoolNonNull, comparison: &compareOperands{left: left, right: right}})
	return admissionResult{route: id}, nil
}

// Binds both ordered references in the same import registry and retains specialized provenance.
func (v *importer) importCompare(route admittedRoute) (occurrenceID, error) {
	if err := v.ensureFields(); err != nil {
		return 0, err
	}
	left, err := importField(v.result.fields, route.comparison.left)
	if err != nil {
		return 0, err
	}
	right, err := importField(v.result.fields, route.comparison.right)
	if err != nil {
		return 0, err
	}
	owner := v.result.admitted.owner
	handle, err := owner.runtime.ComparisonFunction(route.source.GetCompareExpr().Op, left.Type(), right.Type())
	if err != nil {
		return 0, newFailure(TranslationDefect, StageImport, err)
	}
	graph, err := expressions.NewCallExpression(owner.context, handle, []expressions.Expression{left, right})
	if err != nil {
		return 0, newFailure(TranslationDefect, StageImport, err)
	}
	id := appendOccurrence(&v.result.occurrences, route.source, graph, nil)
	v.result.occurrences[id].comparison = route.comparison
	return id, nil
}

// Preserves ordered identities even for symmetric operators and repeated fields.
func (v *graphChecker) checkCompare(id occurrenceID, graph *expressions.CallExpression) error {
	record := v.imported.occurrences[id]
	if err := v.checkComparisonCall(record.imported, graph, record.source.GetCompareExpr().Op); err != nil {
		return err
	}
	if err := checkField(v.imported.fields, record.comparison.left, graph.ChildAt(0), v.stage); err != nil {
		return err
	}
	return checkField(v.imported.fields, record.comparison.right, graph.ChildAt(1), v.stage)
}

// Reconstructs a specialized CompareExpr from checked actual references and operator.
func (v *exporter) exportCompare(id occurrenceID, graph *expressions.CallExpression) (*planpb.Expr, error) {
	record := v.imported.occurrences[id]
	left, err := exportField(v.imported.fields, record.comparison.left, graph.ChildAt(0))
	if err != nil {
		return nil, err
	}
	right, err := exportField(v.imported.fields, record.comparison.right, graph.ChildAt(1))
	if err != nil {
		return nil, err
	}
	op, err := v.imported.admitted.owner.runtime.ComparisonOperator(graph.Handle())
	if err != nil {
		return nil, newFailure(TranslationDefect, StageExport, err)
	}
	return &planpb.Expr{IsTemplate: record.source.IsTemplate, Expr: &planpb.Expr_CompareExpr{CompareExpr: &planpb.CompareExpr{LeftColumnInfo: left, RightColumnInfo: right, Op: op}}}, nil
}

// Independently checks the emitted envelope, operator and both field roles.
func (v *candidateChecker) checkCompare(record occurrence, candidate *planpb.Expr) error {
	if candidate == nil || len(candidate.ProtoReflect().GetUnknown()) != 0 || candidate.IsTemplate != record.source.IsTemplate {
		return translationDefect(StageCandidateValidation, "field comparison changed envelope")
	}
	node, ok := candidate.Expr.(*planpb.Expr_CompareExpr)
	if !ok || node == nil || node.CompareExpr == nil {
		return translationDefect(StageCandidateValidation, "field comparison changed form")
	}
	n := node.CompareExpr
	if len(n.ProtoReflect().GetUnknown()) != 0 || n.Op != record.source.GetCompareExpr().Op {
		return translationDefect(StageCandidateValidation, "field comparison changed operator or metadata")
	}
	if err := checkFieldCandidate(record.comparison.left, n.LeftColumnInfo); err != nil {
		return err
	}
	return checkFieldCandidate(record.comparison.right, n.RightColumnInfo)
}

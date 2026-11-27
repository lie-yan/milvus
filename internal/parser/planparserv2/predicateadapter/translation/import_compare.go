package translation

import (
	"github.com/lie-yan/predicate-optimization/expressions"

	"github.com/milvus-io/milvus/pkg/v3/proto/planpb"
)

// Temporary ordered field operands used during import.
type compareOperands struct {
	// Left reference identity and Type.
	left fieldOperand
	// Right reference identity and Type.
	right fieldOperand
}

// Admits same-domain scalar comparisons without casts or evaluation.
func (v *importer) readCompareOperands(node *planpb.Expr_CompareExpr) (*compareOperands, error) {
	if node == nil || node.CompareExpr == nil {
		return nil, sourceDefect("missing field comparison")
	}
	n := node.CompareExpr
	if n.LeftColumnInfo == nil || n.RightColumnInfo == nil {
		return nil, sourceDefect("missing comparison field")
	}
	if len(n.ProtoReflect().GetUnknown()) != 0 {
		return nil, unsupportedSource(excludedScopeOrVersion)
	}
	if n.Op == planpb.OpType_Invalid {
		return nil, sourceDefect("invalid field comparison operator")
	}
	switch n.Op {
	case planpb.OpType_Equal, planpb.OpType_NotEqual, planpb.OpType_LessThan, planpb.OpType_LessEqual, planpb.OpType_GreaterThan, planpb.OpType_GreaterEqual:
	default:
		return nil, unsupportedSource(unsupportedForm)
	}
	left, leftReason, err := v.result.owner.readFieldOperand(n.LeftColumnInfo)
	if err != nil {
		return nil, err
	}
	// Inspect both fields so an unsupported left domain cannot hide an invalid right field.
	right, rightReason, err := v.result.owner.readFieldOperand(n.RightColumnInfo)
	if err != nil {
		return nil, err
	}
	if leftReason != noLimitation {
		return nil, unsupportedSource(leftReason)
	}
	if rightReason != noLimitation {
		return nil, unsupportedSource(rightReason)
	}
	if !left.typ.EqualsToIgnoringNullability(right.typ) {
		return nil, unsupportedSource(excludedDomain)
	}
	if reason := comparisonDomain(left.typ, n.Op); reason != noLimitation {
		return nil, unsupportedSource(reason)
	}
	return &compareOperands{left: left, right: right}, nil
}

// Binds ordered references and constructs their comparison.
func (v *importer) importCompare(node *planpb.Expr_CompareExpr) (expressions.Expression, error) {
	operands, err := v.readCompareOperands(node)
	if err != nil {
		return nil, err
	}
	if err := v.ensureFields(); err != nil {
		return nil, err
	}
	left, err := importField(v.result.fields, operands.left)
	if err != nil {
		return nil, err
	}
	right, err := importField(v.result.fields, operands.right)
	if err != nil {
		return nil, err
	}
	owner := v.result.owner
	handle, err := owner.runtime.ComparisonFunction(node.CompareExpr.Op, left.Type(), right.Type())
	if err != nil {
		return nil, newFailure(TranslationDefect, StageImport, err)
	}
	graph, err := expressions.NewCallExpression(owner.context, handle, []expressions.Expression{left, right})
	if err != nil {
		return nil, newFailure(TranslationDefect, StageImport, err)
	}
	return graph, nil
}

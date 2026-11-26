package translation

import (
	"github.com/lie-yan/predicate-optimization/expressions"
	"github.com/lie-yan/predicate-optimization/foundation"

	"github.com/milvus-io/milvus/pkg/v3/proto/planpb"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
)

// Native logical form selected from one source node.
type logicalKind uint8

const (
	trueKind logicalKind = iota + 1
	notKind
	andKind
	orKind
)

// Checks the local native shape without visiting children or selecting fallback.
func logicalShape(source *planpb.Expr) (logicalKind, []*planpb.Expr, limitation, error) {
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
		return trueKind, nil, noLimitation, nil
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
		return notKind, []*planpb.Expr{n.Child}, noLimitation, nil
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
			return andKind, []*planpb.Expr{n.Left, n.Right}, noLimitation, nil
		case planpb.BinaryExpr_LogicalOr:
			return orKind, []*planpb.Expr{n.Left, n.Right}, noLimitation, nil
		case planpb.BinaryExpr_Invalid:
			break
		default:
			return 0, nil, unsupportedForm, nil
		}
	}
	return 0, nil, noLimitation, merr.WrapErrServiceInternalMsg("predicate translation: malformed native Boolean expression")
}

// Maps native logical forms; TRUE is constructed as a literal.
func logicalOperator(kind logicalKind) foundation.LogicalOperator {
	switch kind {
	case notKind:
		return foundation.Not
	case andKind:
		return foundation.And
	case orKind:
		return foundation.Or
	default:
		return 0
	}
}

// Constructs ordered native nodes without folding, flattening or evaluating children.
func (v *importer) importLogical(source *planpb.Expr) (expressions.Expression, error) {
	kind, sources, reason, err := logicalShape(source)
	if err != nil {
		return nil, newFailure(UnsafeSource, StageImport, err)
	}
	if reason != noLimitation {
		return nil, unsupportedSource(reason)
	}
	graphs := make([]expressions.Expression, 0, len(sources))
	for _, child := range sources {
		childGraph, err := v.importExpr(child)
		if err != nil {
			return nil, err
		}
		if !v.result.owner.primitives.BoolNonNull.EqualsTo(childGraph.Type()) {
			return nil, unsupportedSource(excludedDomain)
		}
		graphs = append(graphs, childGraph)
	}
	var graph expressions.Expression
	if kind == trueKind {
		value, valueErr := v.result.owner.factory.NewBool(v.result.owner.primitives.BoolNonNull, true)
		if valueErr != nil {
			return nil, newFailure(TranslationDefect, StageImport, valueErr)
		}
		graph, err = expressions.NewLiteralExpression(value)
	} else {
		graph, err = expressions.NewLogicalExpression(logicalOperator(kind), graphs)
	}
	if err != nil {
		return nil, newFailure(TranslationDefect, StageImport, err)
	}
	return graph, nil
}

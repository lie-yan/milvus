package translation

import (
	"context"

	"github.com/lie-yan/predicate-optimization/expressions"

	"github.com/milvus-io/milvus/pkg/v3/proto/planpb"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
)

// Per-import construction state; failures discard all partial graph and registry state.
type importer struct {
	// Cancellation for this invocation.
	ctx context.Context
	// Private graph and source associations under construction.
	result *ImportResult
}

// Capability limitations are distinct from source defects and operational errors.
type limitation uint8

const (
	noLimitation limitation = iota
	unsupportedForm
	unknownResultType
	excludedScopeOrVersion
	excludedDomain
)

// Dispatches only approved forms; excluded regions are not inferred from their spelling.
func (v *importer) importExpr(source *planpb.Expr) (expressions.Expression, error) {
	if err := v.ctx.Err(); err != nil {
		return nil, err
	}
	if source == nil || source.Expr == nil {
		return nil, sourceDefect("missing expression payload")
	}
	if len(source.ProtoReflect().GetUnknown()) != 0 {
		return nil, unsupportedSource(excludedScopeOrVersion)
	}
	switch node := source.Expr.(type) {
	case *planpb.Expr_AlwaysTrueExpr, *planpb.Expr_UnaryExpr, *planpb.Expr_BinaryExpr:
		return v.importLogical(source)
	case *planpb.Expr_TermExpr:
		return v.importTerm(node)
	case *planpb.Expr_BinaryRangeExpr:
		return v.importBinaryRange(node)
	case *planpb.Expr_CompareExpr:
		return v.importCompare(node)
	case *planpb.Expr_UnaryRangeExpr:
		return v.importUnaryRange(node)
	case *planpb.Expr_NullExpr:
		return v.importOpaque(source, node)
	case *planpb.Expr_CallExpr:
		return nil, unsupportedSource(unknownResultType)
	default:
		return nil, unsupportedSource(unsupportedForm)
	}
}

// Classifies unsupported coverage separately from invalid source and internal failures.
func unsupportedSource(reason limitation) error {
	return newFailure(UnsupportedExpression, StageImport, merr.Wrapf(merr.ErrServiceUnimplemented, "predicate translation: import limitation %d", reason))
}

// Classifies malformed known source data without converting it into coverage.
func sourceDefect(message string) error {
	return newFailure(UnsafeSource, StageImport, merr.WrapErrServiceInternalMsg("predicate translation: %s", message))
}

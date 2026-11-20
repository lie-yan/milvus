package translation

import (
	"context"

	"github.com/lie-yan/predicate-optimization/types"

	"github.com/milvus-io/milvus/pkg/v3/proto/planpb"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
)

// Position in one successful admission's routing table.
type routeID int

// Selected paired mapping, independent of protobuf pointer identity.
type routeKind uint8

const (
	opaqueNullRoute routeKind = iota + 1
	trueRoute
	notRoute
	andRoute
	orRoute
	unaryRangeRoute
	compareRoute
	binaryRangeRoute
	termRoute
)

// Compact approved source position; opaque routes have no exposed children.
type admittedRoute struct {
	// Borrowed source region, immutable throughout the attempt.
	source *planpb.Expr
	// Exact paired handler selected by admission.
	kind routeKind
	// Established immutable result Type.
	typ types.Type
	// Ordered exposed child routes; opaque regions remain terminal.
	children []routeID
	// Admitted embedded operands for a unary comparison, otherwise nil.
	unary *unaryRangeOperands
	// Ordered field roles for field-to-field comparison, otherwise nil.
	comparison *compareOperands
	// Field and ordered bound roles for a specialized binary range.
	binaryRange *binaryRangeOperands
	// Target and ordered candidate roles for scalar membership.
	term *termOperands
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

// Either an approved route or a capability limitation that an enclosing policy may inspect.
type admissionResult struct {
	// Approved table position when no limitation is present.
	route routeID
	// Reason no route could be established, never a source-integrity error.
	limitation limitation
}

// Per-invocation source traversal; no graph or registry allocation occurs here.
type admitter struct {
	// Request context used only during this invocation.
	ctx context.Context
	// Pinned semantic and schema inputs.
	translator *Translator
	// Selected source routes, discarded on unsuccessful admission.
	routes []admittedRoute
}

// Dispatches only approved forms; excluded regions are not inferred from their spelling.
func (a *admitter) admitExpr(source *planpb.Expr) (admissionResult, error) {
	if err := a.ctx.Err(); err != nil {
		return admissionResult{}, err
	}
	if source == nil || source.Expr == nil {
		return admissionResult{}, sourceDefect("missing expression payload")
	}
	if len(source.ProtoReflect().GetUnknown()) != 0 {
		return admissionResult{limitation: excludedScopeOrVersion}, nil
	}
	switch node := source.Expr.(type) {
	case *planpb.Expr_AlwaysTrueExpr, *planpb.Expr_UnaryExpr, *planpb.Expr_BinaryExpr:
		return a.admitLogical(source)
	case *planpb.Expr_TermExpr:
		return a.admitTerm(source, node)
	case *planpb.Expr_BinaryRangeExpr:
		return a.admitBinaryRange(source, node)
	case *planpb.Expr_CompareExpr:
		return a.admitCompare(source, node)
	case *planpb.Expr_UnaryRangeExpr:
		return a.admitUnaryRange(source, node)
	case *planpb.Expr_NullExpr:
		return a.admitNull(source, node)
	case *planpb.Expr_CallExpr:
		return admissionResult{limitation: unknownResultType}, nil
	default:
		return admissionResult{limitation: unsupportedForm}, nil
	}
}

// Classifies malformed known source data without converting it into coverage.
func sourceDefect(message string) error {
	return newFailure(UnsafeSource, StageAdmission, merr.WrapErrServiceInternalMsg("predicate translation: %s", message))
}

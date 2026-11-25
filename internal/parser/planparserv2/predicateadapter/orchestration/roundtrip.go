package orchestration

import (
	"context"
	"errors"

	"github.com/lie-yan/predicate-optimization/expressions"

	adapterruntime "github.com/milvus-io/milvus/internal/parser/planparserv2/predicateadapter/runtime"
	"github.com/milvus-io/milvus/internal/parser/planparserv2/predicateadapter/translation"
	"github.com/milvus-io/milvus/pkg/v3/proto/planpb"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
)

// Translates one post-legacy predicate and transfers only a complete accepted candidate.
// Borrowed source/schema must stay immutable; the expression validator must synchronously read without mutation or retention.
func RoundTrip(ctx context.Context, input AttemptInput, validateExpr func(*planpb.Expr) error) (AttemptResult, error) {
	return roundTrip(ctx, input, validateExpr, adapterruntime.SharedRuntime, newTranslator)
}

// Minimal private seam for isolated stage faults; production always uses the paired translator.
type pairedTranslator interface {
	Admit(context.Context, *planpb.Expr) (*translation.AdmissionPlan, error)
	Import(context.Context, *translation.AdmissionPlan) (*translation.ImportResult, error)
	Export(context.Context, *translation.ImportResult, expressions.Expression) (*planpb.Expr, error)
	ValidateCandidate(context.Context, *translation.ImportResult, *planpb.Expr) error
}

// Builds from the acquired runtime without schema refresh.
func newTranslator(input AttemptInput, runtime *adapterruntime.Runtime) (pairedTranslator, error) {
	return translation.New(runtime, input.SchemaView, runtime.Types())
}

// Shares the production sequence with tests without mutable global hooks.
func roundTrip(ctx context.Context, input AttemptInput, validateExpr func(*planpb.Expr) error,
	load func() (*adapterruntime.Runtime, error),
	build func(AttemptInput, *adapterruntime.Runtime) (pairedTranslator, error),
) (AttemptResult, error) {
	if proceed, result, err := checkGate(ctx, input, validateExpr); !proceed {
		return result, err
	}
	report := Report{}
	runtime, err := load()
	if cancelled := ctx.Err(); cancelled != nil {
		return newAbortResult(StageSetup, report, cancelled)
	}
	if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) || IsAdapterAbort(err) {
		return newAbortResult(StageSetup, report, err)
	}
	if err != nil {
		return newRetainedResult(OutcomeRuntimeUnavailable, StageSetup, report, err)
	}
	if runtime == nil {
		return newAbortResult(StageSetup, report, merr.WrapErrServiceInternalMsg("predicate orchestration: runtime loader returned no runtime"))
	}
	translator, err := build(input, runtime)
	if err != nil {
		return newFailureResult(ctx, StageSetup, report, err, false)
	}
	// The caller supplies an already validated original under immutable borrowing.
	// Explicit unsafe-source failures still abort, irrespective of the retention guarantee.
	admitted, err := translator.Admit(ctx, input.Original)
	if err != nil {
		return newFailureResult(ctx, StageAdmission, report, err, true)
	}
	imported, err := translator.Import(ctx, admitted)
	if err != nil {
		return newFailureResult(ctx, StageImport, report, err, true)
	}
	// Export includes framework and adapter graph checks over the actual imported graph.
	candidate, err := translator.Export(ctx, imported, imported.Root())
	if err != nil {
		return newFailureResult(ctx, StageExport, report, err, true)
	}
	if err := translator.ValidateCandidate(ctx, imported, candidate); err != nil {
		return newFailureResult(ctx, StageCandidateValidation, report, err, true)
	}
	if err := ctx.Err(); err != nil {
		return newAbortResult(StagePlacement, report, err)
	}
	if err := validateExpr(candidate); err != nil {
		return newFailureResult(ctx, StagePlacement, report, err, true)
	}
	if err := ctx.Err(); err != nil {
		return newAbortResult(StageAcceptance, report, err)
	}
	result, err := newTranslatedResult(candidate, report)
	if err != nil {
		return newFailureResult(ctx, StageAcceptance, report, err, false)
	}
	return result, nil
}

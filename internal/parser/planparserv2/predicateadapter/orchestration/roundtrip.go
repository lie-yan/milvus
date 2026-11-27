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
	return roundTrip(ctx, input, validateExpr, adapterruntime.SharedRuntime, func(_ context.Context, root expressions.Expression, _ *adapterruntime.Runtime) (expressions.Expression, error) {
		return root, nil
	}, newTranslator)
}

// Rewrites a translated predicate before exporting and validating the complete candidate.
func Optimize(ctx context.Context, input AttemptInput, validateExpr func(*planpb.Expr) error) (AttemptResult, error) {
	return roundTrip(ctx, input, validateExpr, adapterruntime.SharedRuntime, rewrite, newTranslator)
}

// Minimal private seam for isolated stage faults; production always uses the paired translator.
type pairedTranslator interface {
	Import(context.Context, *planpb.Expr) (*translation.ImportResult, error)
	Export(context.Context, *translation.ImportResult, expressions.Expression) (*planpb.Expr, error)
}

// Builds from the acquired runtime without schema refresh.
func newTranslator(input AttemptInput, runtime *adapterruntime.Runtime) (pairedTranslator, error) {
	return translation.New(runtime, input.SchemaView, runtime.Types())
}

// Shares the production sequence with tests without mutable global hooks.
func roundTrip(ctx context.Context, input AttemptInput, validateExpr func(*planpb.Expr) error,
	load func() (*adapterruntime.Runtime, error),
	optimize func(context.Context, expressions.Expression, *adapterruntime.Runtime) (expressions.Expression, error),
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
	if err != nil {
		if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) || IsAdapterAbort(err) {
			return newAbortResult(StageSetup, report, err)
		}
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
	imported, err := translator.Import(ctx, input.Original)
	if err != nil {
		return newFailureResult(ctx, StageImport, report, err, true)
	}
	root, err := optimize(ctx, imported.Root(), runtime)
	if err != nil {
		return newFailureResult(ctx, StageRewrite, report, err, true)
	}
	// Export follows the rewritten graph using the original immutable bindings.
	candidate, err := translator.Export(ctx, imported, root)
	if err != nil {
		return newFailureResult(ctx, StageExport, report, err, true)
	}
	if err := ctx.Err(); err != nil {
		return newAbortResult(StageValidate, report, err)
	}
	if err := validateExpr(candidate); err != nil {
		return newFailureResult(ctx, StageValidate, report, err, true)
	}
	if err := ctx.Err(); err != nil {
		return newAbortResult(StageValidate, report, err)
	}
	result, err := newTranslatedResult(candidate, report)
	if err != nil {
		return newFailureResult(ctx, StageValidate, report, err, false)
	}
	return result, nil
}

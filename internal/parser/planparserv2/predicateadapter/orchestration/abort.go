package orchestration

import (
	"context"
	"errors"

	"github.com/milvus-io/milvus/internal/parser/planparserv2/predicateadapter/translation"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
)

// Private provenance for failures that must escape the optional adapter attempt.
type adapterAbort struct {
	// Original error chain, unchanged by the provenance marker.
	cause error
}

// Adds diagnostic context without replacing the underlying cause.
func (e *adapterAbort) Error() string { return "predicate adapter abort: " + e.cause.Error() }

// Preserves typed errors and cancellation through ordinary wrapping.
func (e *adapterAbort) Unwrap() error { return e.cause }

// Recognizes only adapter-owned abort provenance, including through outer wrappers.
func IsAdapterAbort(err error) bool {
	var abort *adapterAbort
	return errors.As(err, &abort)
}

// Produces no candidate and marks the failure without assigning a retention outcome.
func newAbortResult(stage Stage, report Report, cause error) (AttemptResult, error) {
	report.Outcome, report.Stage, report.Cause = OutcomeUnknown, stage, cause
	return AttemptResult{report: report}, &adapterAbort{cause: cause}
}

// Resolves a failed enabled attempt; originalSafe must be established by the owning operation.
// A stage name alone is insufficient evidence that the original can be retained.
func newFailureResult(ctx context.Context, stage Stage, report Report, cause error, originalSafe bool) (AttemptResult, error) {
	if err := ctx.Err(); err != nil {
		return newAbortResult(stage, report, err)
	}
	if cause == nil {
		return newAbortResult(stage, report, merr.WrapErrServiceInternalMsg("predicate orchestration: missing failure cause"))
	}
	if errors.Is(cause, context.Canceled) || errors.Is(cause, context.DeadlineExceeded) || IsAdapterAbort(cause) {
		return newAbortResult(stage, report, cause)
	}
	outcome := OutcomeUnknown
	var failure *translation.Failure
	if errors.As(cause, &failure) {
		mapped, known := stageFromTranslation(failure.Stage())
		if !known {
			return newAbortResult(stage, report, cause)
		}
		stage = mapped
		switch failure.Kind() {
		case translation.UnsupportedExpression:
			if stage == StageImport || stage == StageExport {
				outcome = OutcomeUnsupported
			}
		case translation.TranslationDefect:
			if stage == StageImport || stage == StageExport {
				outcome = OutcomeRetainedAfterFailure
			}
		}
	} else if stage == StagePlacement {
		// Only the trusted read-only candidate validator supplies errors at this boundary.
		outcome = OutcomeRetainedAfterFailure
	}
	// Cancellation wins over retention even if it arrived during classification.
	if err := ctx.Err(); err != nil {
		return newAbortResult(stage, report, err)
	}
	if !originalSafe || outcome == OutcomeUnknown {
		return newAbortResult(stage, report, cause)
	}
	return newRetainedResult(outcome, stage, report, cause)
}

// Maps distinct stage enums explicitly rather than relying on their numeric layout.
func stageFromTranslation(stage translation.Stage) (Stage, bool) {
	switch stage {
	case translation.StageImport:
		return StageImport, true
	case translation.StageExport:
		return StageExport, true
	default:
		return StageUnknown, false
	}
}

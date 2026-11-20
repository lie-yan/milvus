package orchestration

import (
	"context"

	"github.com/milvus-io/milvus/pkg/v3/proto/planpb"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
)

// Returns proceed=true only for a live attempt with all required inputs.
// A passing gate is not a translated result; the caller must run the complete operation.
func checkGate(ctx context.Context, input AttemptInput, checkPlacement func(*planpb.Expr) error) (bool, AttemptResult, error) {
	report := Report{}
	if ctx == nil {
		result, err := newAbortResult(StageGate, report, merr.WrapErrServiceInternalMsg("predicate orchestration: missing request context"))
		return false, result, err
	}
	if err := ctx.Err(); err != nil {
		result, err := newAbortResult(StageGate, report, err)
		return false, result, err
	}
	var missing string
	switch {
	case input.Original == nil:
		missing = "source expression"
	case input.SchemaView == nil:
		missing = "schema view"
	case checkPlacement == nil:
		missing = "placement validator"
	}
	if missing != "" {
		result, err := newFailureResult(ctx, StageGate, report,
			merr.WrapErrServiceInternalMsg("predicate orchestration: missing %s", missing), false)
		return false, result, err
	}
	if err := ctx.Err(); err != nil {
		result, err := newAbortResult(StageGate, report, err)
		return false, result, err
	}
	return true, AttemptResult{}, nil
}

package orchestration

import (
	"github.com/milvus-io/milvus/pkg/v3/proto/planpb"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
)

// Disposition of one predicate attempt; aborts travel separately as errors.
type Outcome uint8

const (
	// No completed disposition, including the zero result.
	OutcomeUnknown Outcome = iota
	// Complete candidate accepted for caller ownership.
	OutcomeTranslated
	// Requested integration has no available shared runtime.
	OutcomeRuntimeUnavailable
	// Whole-source coverage could not be established.
	OutcomeUnsupported
	// Private attempt failed while the original remained safe to retain.
	OutcomeRetainedAfterFailure
)

// Operation boundary reached or failed, including stages outside paired translation.
type Stage uint8

const (
	// No operation stage recorded.
	StageUnknown Stage = iota
	// Required input and context checks.
	StageGate
	// Runtime availability and translator setup.
	StageSetup
	// Source coverage, graph construction and attempt-local associations.
	StageImport
	// Fresh candidate reconstruction.
	StageExport
	// Trusted parser placement check.
	StagePlacement
	// Final cancellation check and candidate transfer.
	StageAcceptance
)

// Diagnostic value copy; causes must not retain expressions, schemas or session state.
type Report struct {
	// Translation or specific retention reason.
	Outcome Outcome
	// Last completed or failed operation boundary.
	Stage Stage
	// Original diagnostic error, without reclassification or payload dumps.
	Cause error
}

// Completed disposition with a candidate only on translated success.
// The caller owns accepted output; copying this value does not clone that output.
type AttemptResult struct {
	// Fresh accepted protobuf, absent for retention and zero results.
	candidate *planpb.Expr
	// Diagnostics without borrowed input or graph associations.
	report Report
}

// Returns the disposition; the zero result is never translated.
func (r AttemptResult) Outcome() Outcome { return r.report.Outcome }

// Returns caller-owned output on success, otherwise nil.
func (r AttemptResult) Candidate() *planpb.Expr { return r.candidate }

// Copies diagnostic fields without exposing mutable result storage.
func (r AttemptResult) Report() Report { return r.report }

// Builds retention without any candidate argument or original-expression alias.
func newRetainedResult(outcome Outcome, stage Stage, report Report, cause error) (AttemptResult, error) {
	switch outcome {
	case OutcomeRuntimeUnavailable, OutcomeUnsupported, OutcomeRetainedAfterFailure:
	default:
		return AttemptResult{}, merr.WrapErrServiceInternalMsg("predicate orchestration: invalid retention outcome")
	}
	report.Outcome, report.Stage, report.Cause = outcome, stage, cause
	return AttemptResult{report: report}, nil
}

// Transfers an already accepted candidate; the operation must validate it before calling.
func newTranslatedResult(candidate *planpb.Expr, report Report) (AttemptResult, error) {
	if candidate == nil {
		return AttemptResult{}, merr.WrapErrServiceInternalMsg("predicate orchestration: missing accepted candidate")
	}
	report.Outcome, report.Stage, report.Cause = OutcomeTranslated, StageAcceptance, nil
	return AttemptResult{candidate: candidate, report: report}, nil
}

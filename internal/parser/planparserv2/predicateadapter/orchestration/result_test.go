package orchestration

import (
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/milvus-io/milvus/pkg/v3/proto/planpb"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
)

func TestRetainedResultPreservesDiagnostics(t *testing.T) {
	cause := merr.WrapErrServiceInternalMsg("private candidate defect")
	result, err := newRetainedResult(OutcomeRetainedAfterFailure, StageExport, Report{}, cause)
	require.NoError(t, err)
	require.Nil(t, result.Candidate())
	report := result.Report()
	require.Equal(t, Report{
		Outcome: OutcomeRetainedAfterFailure, Stage: StageExport,
		Cause: cause,
	}, report)
	// Caller edits to diagnostic values cannot change the operation's disposition.
	report.Outcome = OutcomeTranslated
	require.Equal(t, OutcomeRetainedAfterFailure, result.Outcome())
	bad, err := newRetainedResult(OutcomeTranslated, StageValidate, report, nil)
	require.ErrorIs(t, err, merr.ErrServiceInternal)
	require.NotEqual(t, OutcomeTranslated, bad.Outcome())
	require.Nil(t, bad.Candidate())
}

func TestTranslatedResultRequiresCandidate(t *testing.T) {
	result, err := newTranslatedResult(nil, Report{Outcome: OutcomeTranslated})
	require.ErrorIs(t, err, merr.ErrServiceInternal)
	require.NotEqual(t, OutcomeTranslated, result.Outcome())
	require.Nil(t, result.Candidate())

	candidate := &planpb.Expr{Expr: &planpb.Expr_AlwaysTrueExpr{AlwaysTrueExpr: &planpb.AlwaysTrueExpr{}}}
	result, err = newTranslatedResult(candidate, Report{Cause: merr.ErrServiceInternal})
	require.NoError(t, err)
	require.Equal(t, OutcomeTranslated, result.Outcome())
	require.Same(t, candidate, result.Candidate(), "acceptance transfers output without copying")
	require.Equal(t, Report{Outcome: OutcomeTranslated, Stage: StageValidate}, result.Report())
}

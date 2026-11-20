package orchestration

import (
	"context"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
	adapterruntime "github.com/milvus-io/milvus/internal/parser/planparserv2/predicateadapter/runtime"
	"github.com/milvus-io/milvus/internal/parser/planparserv2/predicateadapter/schema"
	"github.com/milvus-io/milvus/internal/parser/planparserv2/predicateadapter/translation"
	"github.com/milvus-io/milvus/pkg/v3/proto/planpb"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
	"github.com/milvus-io/milvus/pkg/v3/util/typeutil"
)

func TestTranslationFailureDisposition(t *testing.T) {
	ctx := context.Background()
	runtime, err := adapterruntime.SharedRuntime()
	require.NoError(t, err)
	helper, err := typeutil.CreateSchemaHelper(&schemapb.CollectionSchema{})
	require.NoError(t, err)
	view, err := schema.NewView(helper, nil)
	require.NoError(t, err)
	translator, err := translation.New(runtime, view, runtime.Types())
	require.NoError(t, err)
	_, coverage := translator.Admit(ctx, &planpb.Expr{Expr: &planpb.Expr_CallExpr{CallExpr: &planpb.CallExpr{FunctionName: "unknown"}}})
	_, unsafe := translator.Admit(ctx, &planpb.Expr{})
	_, importDefect := translator.Import(ctx, nil)
	source := &planpb.Expr{Expr: &planpb.Expr_AlwaysTrueExpr{AlwaysTrueExpr: &planpb.AlwaysTrueExpr{}}}
	admitted, err := translator.Admit(ctx, source)
	require.NoError(t, err)
	imported, err := translator.Import(ctx, admitted)
	require.NoError(t, err)
	_, graphDefect := translator.Export(ctx, imported, nil)
	_, exportDefect := translator.Export(ctx, nil, imported.Root())
	translationDefect := translator.ValidateCandidate(ctx, imported, &planpb.Expr{})
	for _, tc := range []struct {
		name    string
		cause   error
		stage   Stage
		outcome Outcome
	}{
		{"coverage", coverage, StageAdmission, OutcomeUnsupported},
		{"unsafe source", unsafe, StageAdmission, OutcomeUnknown},
		{"import", importDefect, StageImport, OutcomeRetainedAfterFailure},
		{"graph", graphDefect, StageExport, OutcomeRetainedAfterFailure},
		{"export", exportDefect, StageExport, OutcomeRetainedAfterFailure},
		{"candidate", translationDefect, StageCandidateValidation, OutcomeRetainedAfterFailure},
	} {
		t.Run(tc.name, func(t *testing.T) {
			require.Error(t, tc.cause)
			wrapped := merr.Wrap(tc.cause, "translation stage")
			result, err := newFailureResult(ctx, StageUnknown, Report{}, wrapped, true)
			require.Equal(t, tc.outcome, result.Outcome())
			require.Equal(t, tc.stage, result.Report().Stage)
			require.Same(t, wrapped, result.Report().Cause)
			require.Nil(t, result.Candidate())
			if tc.outcome == OutcomeUnknown {
				require.True(t, IsAdapterAbort(err))
				require.ErrorIs(t, err, tc.cause)
			} else {
				require.NoError(t, err)
			}
		})
	}
	// A candidate defect cannot justify retention when source integrity is not guaranteed.
	result, err := newFailureResult(ctx, StageExport, Report{}, exportDefect, false)
	require.True(t, IsAdapterAbort(err))
	require.Nil(t, result.Candidate())
	require.Equal(t, OutcomeUnknown, result.Outcome())
}

func TestAbortProvenanceAndPlacement(t *testing.T) {
	ctx := context.Background()
	cause := merr.WrapErrServiceInternalMsg("unclassified setup failure")
	result, err := newFailureResult(ctx, StageSetup, Report{}, cause, true)
	require.Nil(t, result.Candidate())
	wrapped := merr.Wrap(err, "scorer context")
	require.True(t, IsAdapterAbort(wrapped))
	require.ErrorIs(t, wrapped, cause)
	require.Equal(t, merr.Code(cause), merr.Code(wrapped))
	require.False(t, IsAdapterAbort(merr.Wrap(cause, "ordinary parser context")))

	placement := merr.WrapErrQueryPlanMsg("candidate placement rejected")
	result, err = newFailureResult(ctx, StagePlacement, Report{}, placement, true)
	require.NoError(t, err)
	require.Equal(t, OutcomeRetainedAfterFailure, result.Outcome())
	require.Same(t, placement, result.Report().Cause)
	require.Nil(t, result.Candidate())
}

// Becomes cancelled at the final disposition check without timing-dependent goroutines.
type dispositionCancellation struct {
	context.Context
	checks int
}

func (c *dispositionCancellation) Err() error {
	c.checks++
	if c.checks >= 2 {
		return context.Canceled
	}
	return nil
}

func TestFailureCancellationPrecedence(t *testing.T) {
	for _, cause := range []error{context.Canceled, context.DeadlineExceeded} {
		// A callback can return cancellation before the request context reports it.
		wrapped := merr.Wrap(cause, "placement callback")
		result, err := newFailureResult(context.Background(), StagePlacement, Report{}, wrapped, true)
		require.True(t, IsAdapterAbort(err))
		require.ErrorIs(t, err, cause)
		require.Same(t, wrapped, result.Report().Cause)
		require.Equal(t, OutcomeUnknown, result.Outcome())
		require.Nil(t, result.Candidate())
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	for _, ctx := range []context.Context{ctx, &dispositionCancellation{Context: context.Background()}} {
		result, err := newFailureResult(ctx, StagePlacement, Report{}, merr.ErrQueryPlan, true)
		require.True(t, IsAdapterAbort(err))
		require.ErrorIs(t, err, context.Canceled)
		require.Nil(t, result.Candidate())
		require.Equal(t, OutcomeUnknown, result.Outcome())
	}
}

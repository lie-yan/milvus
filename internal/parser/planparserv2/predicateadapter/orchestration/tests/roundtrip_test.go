package orchestration_test

import (
	"context"
	"testing"

	"github.com/stretchr/testify/require"
	"google.golang.org/protobuf/proto"

	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
	"github.com/milvus-io/milvus/internal/parser/planparserv2/predicateadapter/orchestration"
	"github.com/milvus-io/milvus/internal/parser/planparserv2/predicateadapter/schema"
	"github.com/milvus-io/milvus/pkg/v3/proto/planpb"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
	"github.com/milvus-io/milvus/pkg/v3/util/typeutil"
)

// Native comparison beside an approved opaque null predicate.
func roundTripInput(t *testing.T) (orchestration.AttemptInput, *schemapb.CollectionSchema) {
	t.Helper()
	fields := &schemapb.CollectionSchema{Fields: []*schemapb.FieldSchema{
		{FieldID: 101, Name: "i", DataType: schemapb.DataType_Int64},
		{FieldID: 102, Name: "b", DataType: schemapb.DataType_Bool, Nullable: true},
	}}
	helper, err := typeutil.CreateSchemaHelper(fields)
	require.NoError(t, err)
	view, err := schema.NewView(helper, fields.Fields)
	require.NoError(t, err)
	source := &planpb.Expr{Expr: &planpb.Expr_BinaryExpr{BinaryExpr: &planpb.BinaryExpr{
		Op: planpb.BinaryExpr_LogicalAnd,
		Left: &planpb.Expr{Expr: &planpb.Expr_UnaryRangeExpr{UnaryRangeExpr: &planpb.UnaryRangeExpr{
			ColumnInfo: &planpb.ColumnInfo{FieldId: 101, DataType: schemapb.DataType_Int64},
			Op:         planpb.OpType_GreaterThan, Value: &planpb.GenericValue{Val: &planpb.GenericValue_Int64Val{Int64Val: 7}},
		}}},
		Right: &planpb.Expr{Expr: &planpb.Expr_NullExpr{NullExpr: &planpb.NullExpr{
			ColumnInfo: &planpb.ColumnInfo{FieldId: 102, DataType: schemapb.DataType_Bool, Nullable: true}, Op: planpb.NullExpr_IsNull,
		}}},
	}}}
	return orchestration.AttemptInput{Original: source, SchemaView: view}, fields
}

func TestRoundTripAcceptanceAndOwnership(t *testing.T) {
	input, fields := roundTripInput(t)
	sourceBefore, schemaBefore := proto.Clone(input.Original), proto.Clone(fields)
	calls := 0
	result, err := orchestration.RoundTrip(context.Background(), input, func(candidate *planpb.Expr) error {
		calls++
		require.NotSame(t, input.Original, candidate)
		require.True(t, proto.Equal(input.Original, candidate), "placement must receive the complete specialized output")
		return nil
	})
	require.NoError(t, err)
	require.Equal(t, 1, calls)
	require.Equal(t, orchestration.OutcomeTranslated, result.Outcome())
	require.Equal(t, orchestration.StageAcceptance, result.Report().Stage)
	candidate := result.Candidate()
	candidate.GetBinaryExpr().Left.GetUnaryRangeExpr().Value.Val = &planpb.GenericValue_Int64Val{Int64Val: 999}
	candidate.GetBinaryExpr().Right.GetNullExpr().ColumnInfo.FieldId = 999
	require.True(t, proto.Equal(sourceBefore, input.Original))
	require.True(t, proto.Equal(schemaBefore, fields))
}

func TestRoundTripAdmissionOutcomes(t *testing.T) {
	input, _ := roundTripInput(t)
	for _, tc := range []struct {
		name    string
		right   *planpb.Expr
		outcome orchestration.Outcome
	}{
		{"unsupported descendant", &planpb.Expr{Expr: &planpb.Expr_CallExpr{CallExpr: &planpb.CallExpr{FunctionName: "unknown"}}}, orchestration.OutcomeUnsupported},
		{"unsafe descendant", &planpb.Expr{}, orchestration.OutcomeUnknown},
	} {
		t.Run(tc.name, func(t *testing.T) {
			in := input
			in.Original = proto.Clone(input.Original).(*planpb.Expr)
			in.Original.GetBinaryExpr().Right = tc.right
			before := proto.Clone(in.Original)
			result, err := orchestration.RoundTrip(context.Background(), in, func(*planpb.Expr) error { t.Fatal("admission failure reached placement"); return nil })
			require.Equal(t, tc.outcome, result.Outcome())
			require.Equal(t, orchestration.StageAdmission, result.Report().Stage)
			require.Nil(t, result.Candidate())
			require.True(t, proto.Equal(before, in.Original))
			if tc.outcome == orchestration.OutcomeUnsupported {
				require.NoError(t, err)
			} else {
				require.True(t, orchestration.IsAdapterAbort(err))
				require.ErrorIs(t, err, merr.ErrServiceInternal)
			}
		})
	}

}

func TestRoundTripPlacementFailuresAndCancellation(t *testing.T) {
	input, _ := roundTripInput(t)
	before := proto.Clone(input.Original)
	placementErr := merr.WrapErrQueryPlanMsg("candidate placement rejected")
	for _, tc := range []struct {
		name    string
		cause   error
		cancel  bool
		outcome orchestration.Outcome
		stage   orchestration.Stage
	}{
		{"rejected", placementErr, false, orchestration.OutcomeRetainedAfterFailure, orchestration.StagePlacement},
		{"callback cancellation", merr.Wrap(context.Canceled, "validator"), false, orchestration.OutcomeUnknown, orchestration.StagePlacement},
		{"callback deadline", merr.Wrap(context.DeadlineExceeded, "validator"), false, orchestration.OutcomeUnknown, orchestration.StagePlacement},
		{"cancel before transfer", nil, true, orchestration.OutcomeUnknown, orchestration.StageAcceptance},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			calls := 0
			result, err := orchestration.RoundTrip(ctx, input, func(candidate *planpb.Expr) error {
				calls++
				require.True(t, proto.Equal(input.Original, candidate))
				if tc.cancel {
					cancel()
				}
				return tc.cause
			})
			require.Equal(t, 1, calls)
			require.Equal(t, tc.outcome, result.Outcome())
			require.Equal(t, tc.stage, result.Report().Stage)
			require.Nil(t, result.Candidate())
			require.True(t, proto.Equal(before, input.Original))
			if tc.outcome == orchestration.OutcomeRetainedAfterFailure {
				require.NoError(t, err)
				require.Same(t, tc.cause, result.Report().Cause)
			} else {
				require.True(t, orchestration.IsAdapterAbort(err))
				if tc.cancel {
					require.ErrorIs(t, err, context.Canceled)
				} else {
					require.ErrorIs(t, err, tc.cause)
				}
			}
		})
	}
}

func TestRoundTripAttemptsRemainIndependent(t *testing.T) {
	input, fields := roundTripInput(t)
	before, schemaBefore := proto.Clone(input.Original), proto.Clone(fields)
	placement := func(*planpb.Expr) error { return nil }
	first, err := orchestration.RoundTrip(context.Background(), input, placement)
	require.NoError(t, err)
	require.Equal(t, orchestration.OutcomeTranslated, first.Outcome())
	second, err := orchestration.RoundTrip(context.Background(), input, placement)
	require.NoError(t, err)
	require.Equal(t, orchestration.OutcomeTranslated, second.Outcome())
	first.Candidate().GetBinaryExpr().Left.GetUnaryRangeExpr().ColumnInfo.FieldId = 999
	first.Candidate().GetBinaryExpr().Right.GetNullExpr().ColumnInfo.FieldId = 999
	require.True(t, proto.Equal(before, second.Candidate()))

	retained, err := orchestration.RoundTrip(context.Background(), input, func(*planpb.Expr) error { return merr.ErrQueryPlan })
	require.NoError(t, err)
	require.Equal(t, orchestration.OutcomeRetainedAfterFailure, retained.Outcome())
	require.Nil(t, retained.Candidate())
	cancelled, cancel := context.WithCancel(context.Background())
	cancel()
	aborted, err := orchestration.RoundTrip(cancelled, input, placement)
	require.True(t, orchestration.IsAdapterAbort(err))
	require.Nil(t, aborted.Candidate())

	last, err := orchestration.RoundTrip(context.Background(), input, placement)
	require.NoError(t, err)
	require.Equal(t, orchestration.OutcomeTranslated, last.Outcome())
	require.True(t, proto.Equal(before, last.Candidate()))
	require.True(t, proto.Equal(before, input.Original))
	require.True(t, proto.Equal(schemaBefore, fields))
}

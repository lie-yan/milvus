package orchestration_test

import (
	"context"
	"sync"
	"testing"

	"github.com/stretchr/testify/require"
	"google.golang.org/protobuf/proto"

	"github.com/milvus-io/milvus/internal/parser/planparserv2/predicateadapter/orchestration"
	"github.com/milvus-io/milvus/pkg/v3/proto/planpb"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
)

func TestConcurrentRoundTripIsolation(t *testing.T) {
	input, fields := roundTripInput(t)
	sourceBefore, schemaBefore := proto.Clone(input.Original), proto.Clone(fields)
	// Empty-pipeline expectation; use configuration-specific candidates when adding passes.
	expected := proto.Clone(input.Original)
	const attempts = 4
	const rejected, cancelled = 2, 3
	results := make([]orchestration.AttemptResult, attempts)
	errors := make([]error, attempts)
	placementCalls := make([]int, attempts)
	placementMatches := make([]bool, attempts)
	start, release := make(chan struct{}), make(chan struct{})
	var ready, arrived, workers sync.WaitGroup
	ready.Add(attempts)
	arrived.Add(attempts)
	for i := range attempts {
		workers.Go(func() {
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			var arrival sync.Once
			// An early failure must release the barrier too, so a regression fails instead of hanging.
			defer arrival.Do(arrived.Done)
			ready.Done()
			<-start
			results[i], errors[i] = orchestration.RoundTrip(ctx, input, func(candidate *planpb.Expr) error {
				placementCalls[i]++
				placementMatches[i] = proto.Equal(expected, candidate)
				arrival.Do(arrived.Done)
				<-release
				switch i {
				case rejected:
					return merr.ErrQueryPlan
				case cancelled:
					cancel()
				}
				return nil
			})
		})
	}
	ready.Wait()
	close(start)
	arrived.Wait()
	close(release)
	workers.Wait()

	for i := range attempts {
		require.Equal(t, 1, placementCalls[i], "attempt %d must reach placement while peers remain active", i)
		require.True(t, placementMatches[i], "attempt %d placement candidate", i)
	}
	for i := range rejected {
		require.NoError(t, errors[i])
		require.Equal(t, orchestration.OutcomeTranslated, results[i].Outcome())
		require.True(t, proto.Equal(expected, results[i].Candidate()))
	}
	require.NoError(t, errors[rejected])
	require.Equal(t, orchestration.OutcomeRetainedAfterFailure, results[rejected].Outcome())
	require.Equal(t, orchestration.StageValidate, results[rejected].Report().Stage)
	require.ErrorIs(t, results[rejected].Report().Cause, merr.ErrQueryPlan)
	require.Nil(t, results[rejected].Candidate())
	require.True(t, orchestration.IsAdapterAbort(errors[cancelled]))
	require.ErrorIs(t, errors[cancelled], context.Canceled)
	require.Equal(t, orchestration.StageValidate, results[cancelled].Report().Stage)
	require.Nil(t, results[cancelled].Candidate())

	// All attempts have returned before any candidate is mutated.
	first, second := results[0].Candidate(), results[1].Candidate()
	require.NotSame(t, first, second)
	first.GetBinaryExpr().Left.GetUnaryRangeExpr().Value.Val = &planpb.GenericValue_Int64Val{Int64Val: 999}
	first.GetBinaryExpr().Left.GetUnaryRangeExpr().ColumnInfo.FieldId = 999
	first.GetBinaryExpr().Right.GetNullExpr().ColumnInfo.FieldId = 999
	require.True(t, proto.Equal(expected, second))
	require.True(t, proto.Equal(sourceBefore, input.Original))
	require.True(t, proto.Equal(schemaBefore, fields))
}

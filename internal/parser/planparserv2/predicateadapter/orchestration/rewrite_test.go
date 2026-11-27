package orchestration

import (
	"context"
	"testing"

	"github.com/lie-yan/predicate-optimization/expressions"
	"github.com/lie-yan/predicate-optimization/foundation"
	"github.com/lie-yan/predicate-optimization/rewriter"
	"github.com/stretchr/testify/require"
	"google.golang.org/protobuf/proto"

	adapterruntime "github.com/milvus-io/milvus/internal/parser/planparserv2/predicateadapter/runtime"
	"github.com/milvus-io/milvus/pkg/v3/proto/planpb"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
)

// Simulates a provider/rule failure after a successful import.
type failingRewrite struct {
	rewriter.BaseExpressionRewriter[struct{}]
	cause error
}

func (r failingRewrite) RewriteLiteral(node *expressions.LiteralExpression, _ struct{}, _ rewriter.ExpressionTreeRewriter[struct{}]) (foundation.Optional[expressions.Expression], error) {
	return foundation.Some[expressions.Expression](node), r.cause
}

func TestRewriteFailureRetainsOriginalAndCancellationAborts(t *testing.T) {
	for _, cause := range []error{merr.ErrServiceInternal, context.Canceled} {
		input := gateInput(t)
		before := proto.Clone(input.Original)
		optimize := func(ctx context.Context, root expressions.Expression, _ *adapterruntime.Runtime) (expressions.Expression, error) {
			return applyRewriters(ctx, root, []rewriter.ExpressionRewriter[struct{}]{failingRewrite{cause: cause}})
		}
		result, err := roundTrip(context.Background(), input, func(*planpb.Expr) error { t.Fatal("failed rewrite reached validation"); return nil }, adapterruntime.SharedRuntime, optimize, newTranslator)
		require.Nil(t, result.Candidate())
		require.Equal(t, StageRewrite, result.Report().Stage)
		require.ErrorIs(t, result.Report().Cause, cause)
		require.True(t, proto.Equal(before, input.Original))
		if cause == context.Canceled {
			require.True(t, IsAdapterAbort(err))
			require.ErrorIs(t, err, cause)
		} else {
			require.NoError(t, err)
			require.Equal(t, OutcomeRetainedAfterFailure, result.Outcome())
		}
	}
}

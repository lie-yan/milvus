package orchestration

import (
	"context"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
	"github.com/milvus-io/milvus/internal/parser/planparserv2/predicateadapter/schema"
	"github.com/milvus-io/milvus/pkg/v3/proto/planpb"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
	"github.com/milvus-io/milvus/pkg/v3/util/typeutil"
)

func gateInput(t *testing.T) AttemptInput {
	t.Helper()
	helper, err := typeutil.CreateSchemaHelper(&schemapb.CollectionSchema{})
	require.NoError(t, err)
	view, err := schema.NewView(helper, nil)
	require.NoError(t, err)
	return AttemptInput{
		Original:   &planpb.Expr{Expr: &planpb.Expr_AlwaysTrueExpr{AlwaysTrueExpr: &planpb.AlwaysTrueExpr{}}},
		SchemaView: view,
	}
}

func TestGateRequiredInputsAndProceed(t *testing.T) {
	input := gateInput(t)
	placement := func(*planpb.Expr) error { t.Fatal("gate invoked placement"); return nil }
	for _, missing := range []string{"context", "source", "schema", "placement"} {
		t.Run(missing, func(t *testing.T) {
			in, ctx, check := input, context.Context(context.Background()), placement
			switch missing {
			case "context":
				ctx = nil
			case "source":
				in.Original = nil
			case "schema":
				in.SchemaView = nil
			case "placement":
				check = nil
			}
			proceed, result, err := checkGate(ctx, in, check)
			require.False(t, proceed)
			require.True(t, IsAdapterAbort(err))
			require.ErrorIs(t, err, merr.ErrServiceInternal)
			require.Nil(t, result.Candidate())
		})
	}
	proceed, result, err := checkGate(context.Background(), input, placement)
	require.NoError(t, err)
	require.True(t, proceed)
	require.NotEqual(t, OutcomeTranslated, result.Outcome())
	require.Nil(t, result.Candidate())
}

func TestGateCancellationPrecedence(t *testing.T) {
	input := gateInput(t)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	for _, in := range []AttemptInput{input, {}} {
		proceed, result, err := checkGate(ctx, in, nil)
		require.False(t, proceed)
		require.True(t, IsAdapterAbort(err))
		require.ErrorIs(t, err, context.Canceled)
		require.Nil(t, result.Candidate())
	}
	proceed, result, err := checkGate(&dispositionCancellation{Context: context.Background()}, input, func(*planpb.Expr) error { return nil })
	require.False(t, proceed)
	require.ErrorIs(t, err, context.Canceled)
	require.Nil(t, result.Candidate())
}

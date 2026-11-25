package integration

import (
	"context"
	"testing"

	"github.com/bytedance/mockey"
	"github.com/stretchr/testify/require"
	"google.golang.org/protobuf/proto"

	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
	"github.com/milvus-io/milvus/internal/parser/planparserv2/predicateadapter/orchestration"
	adapterruntime "github.com/milvus-io/milvus/internal/parser/planparserv2/predicateadapter/runtime"
	"github.com/milvus-io/milvus/internal/parser/planparserv2/predicateadapter/translation"
	"github.com/milvus-io/milvus/pkg/v3/proto/planpb"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
	"github.com/milvus-io/milvus/pkg/v3/util/typeutil"
)

func integrationFixture(t *testing.T) (*typeutil.SchemaHelper, *planpb.Expr) {
	t.Helper()
	helper, err := typeutil.CreateSchemaHelper(&schemapb.CollectionSchema{Fields: []*schemapb.FieldSchema{
		{FieldID: 101, Name: "i", DataType: schemapb.DataType_Int64},
	}})
	require.NoError(t, err)
	original := &planpb.Expr{Expr: &planpb.Expr_UnaryRangeExpr{UnaryRangeExpr: &planpb.UnaryRangeExpr{
		ColumnInfo: &planpb.ColumnInfo{FieldId: 101, DataType: schemapb.DataType_Int64},
		Op:         planpb.OpType_GreaterThan, Value: &planpb.GenericValue{Val: &planpb.GenericValue_Int64Val{Int64Val: 7}},
	}}}
	return helper, original
}

func acceptCandidate(*planpb.Expr) error { return nil }

func TestPredicateOptimizationRetainsOriginal(t *testing.T) {
	helper, native := integrationFixture(t)
	for _, kind := range []string{"unsupported", "runtime unavailable", "translation defect"} {
		t.Run(kind, func(t *testing.T) {
			original := proto.Clone(native).(*planpb.Expr)
			switch kind {
			case "unsupported":
				original = &planpb.Expr{Expr: &planpb.Expr_CallExpr{CallExpr: &planpb.CallExpr{FunctionName: "unknown"}}}
			case "runtime unavailable":
				patch := mockey.Mock(adapterruntime.SharedRuntime).Return(nil, merr.ErrServiceUnavailable).Build()
				defer patch.UnPatch()
			case "translation defect":
				var run func(*translation.Translator, context.Context, *translation.AdmissionPlan) (*translation.ImportResult, error)
				patch := mockey.Mock((*translation.Translator).Import).Origin(&run).To(func(tr *translation.Translator, ctx context.Context, _ *translation.AdmissionPlan) (*translation.ImportResult, error) {
					return run(tr, ctx, nil) // Real classified defect after successful admission.
				}).Build()
				defer patch.UnPatch()
			}
			before := proto.Clone(original)
			got, err := Apply(helper, original, acceptCandidate)
			require.NoError(t, err)
			require.Same(t, original, got)
			require.True(t, proto.Equal(before, original))
		})
	}
}

func TestPredicateOptimizationAborts(t *testing.T) {
	helper, original := integrationFixture(t)
	for _, kind := range []string{"missing helper", "inconsistent membership", "missing original", "unsafe source", "translator setup", "unknown outcome"} {
		t.Run(kind, func(t *testing.T) {
			h, source := helper, original
			cause := merr.ErrServiceInternal
			switch kind {
			case "missing helper":
				h = nil
			case "inconsistent membership":
				// A copied node cannot be paired with the existing helper's identity index.
				fields := proto.Clone(&schemapb.CollectionSchema{Fields: helper.TopLevelFields()}).(*schemapb.CollectionSchema).Fields
				patch := mockey.Mock((*typeutil.SchemaHelper).TopLevelFields).Return(fields).Build()
				defer patch.UnPatch()
			case "missing original":
				source = nil
			case "unsafe source":
				source = &planpb.Expr{}
			case "translator setup":
				cause = merr.ErrServiceUnavailable
				patch := mockey.Mock(translation.New).Return(nil, cause).Build()
				defer patch.UnPatch()
			case "unknown outcome":
				patch := mockey.Mock(orchestration.RoundTrip).Return(orchestration.AttemptResult{}, nil).Build()
				defer patch.UnPatch()
			}
			got, err := Apply(h, source, acceptCandidate)
			require.Nil(t, got)
			require.ErrorIs(t, err, cause)
			require.True(t, IsFailure(merr.Wrap(err, "outer parser caller")))
			if kind == "translator setup" || kind == "unsafe source" || kind == "missing original" {
				require.True(t, orchestration.IsAdapterAbort(err))
			}
		})
	}
	require.False(t, IsFailure(merr.ErrServiceInternal), "error code alone is not provenance")
}

func TestPredicateOptimizationCandidateValidator(t *testing.T) {
	helper, original := integrationFixture(t)
	before := proto.Clone(original)
	for _, reject := range []bool{false, true} {
		calls := 0
		got, err := Apply(helper, original, func(candidate *planpb.Expr) error {
			calls++
			require.NotSame(t, original, candidate)
			require.True(t, proto.Equal(original, candidate))
			if reject {
				return merr.ErrQueryPlan
			}
			return nil
		})
		require.NoError(t, err)
		require.Equal(t, 1, calls)
		if reject {
			require.Same(t, original, got)
		} else {
			require.NotSame(t, original, got)
		}
		require.True(t, proto.Equal(before, original))
	}
	got, err := Apply(helper, original, nil)
	require.Nil(t, got)
	require.True(t, IsFailure(err))
	require.True(t, orchestration.IsAdapterAbort(err))
	require.ErrorIs(t, err, merr.ErrServiceInternal)
}

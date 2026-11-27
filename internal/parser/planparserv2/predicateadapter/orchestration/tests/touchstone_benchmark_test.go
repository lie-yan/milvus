package orchestration_test

import (
	"context"
	"strconv"
	"testing"

	"github.com/stretchr/testify/require"
	"google.golang.org/protobuf/proto"

	"github.com/milvus-io/milvus/internal/parser/planparserv2"
	"github.com/milvus-io/milvus/internal/parser/planparserv2/predicateadapter/orchestration"
	"github.com/milvus-io/milvus/internal/parser/planparserv2/predicateadapter/schema"
	"github.com/milvus-io/milvus/internal/parser/planparserv2/predicateadapter/tests/touchstone"
	"github.com/milvus-io/milvus/pkg/v3/proto/planpb"
	"github.com/milvus-io/milvus/pkg/v3/util/paramtable"
	"github.com/milvus-io/milvus/pkg/v3/util/typeutil"
)

// Measures import/export versus import/rewrite/export, excluding parsing and schema setup.
func BenchmarkTouchstonePipeline(b *testing.B) {
	paramtable.Init()
	params := paramtable.Get()
	for _, item := range []*paramtable.ParamItem{&params.ProxyCfg.EnablePredicateOptimization, &params.CommonCfg.EnabledOptimizeExpr} {
		key, previous := item.Key, item.GetValue()
		require.NoError(b, params.Save(key, "false"))
		b.Cleanup(func() { require.NoError(b, params.Save(key, previous)) })
	}
	collection := touchstone.Schema("benchmark", false)
	helper, err := typeutil.CreateSchemaHelper(collection)
	require.NoError(b, err)
	original, err := planparserv2.ParseExpr(helper, touchstone.Query, nil)
	require.NoError(b, err)
	view, err := schema.NewView(helper, collection.Fields)
	require.NoError(b, err)
	input := orchestration.AttemptInput{Original: original, SchemaView: view}
	for _, tc := range []struct {
		name string
		run  func(context.Context, orchestration.AttemptInput, func(*planpb.Expr) error) (orchestration.AttemptResult, error)
	}{{"empty", orchestration.RoundTrip}, {"five_passes", orchestration.Optimize}} {
		b.Run(tc.name, func(b *testing.B) {
			ctx := context.Background()
			validate := func(*planpb.Expr) error { return nil }
			result, err := tc.run(ctx, input, validate)
			require.NoError(b, err)
			require.Equal(b, orchestration.OutcomeTranslated, result.Outcome())
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				result, err = tc.run(ctx, input, validate)
				if err != nil || result.Outcome() != orchestration.OutcomeTranslated {
					b.Fatalf("pipeline failed: %v, %v", result.Report(), err)
				}
			}
		})
	}
}

// Compares warmed adapter cost and full retrieve-plan construction on fixed workloads.
func BenchmarkPredicatePlanning(b *testing.B) {
	paramtable.Init()
	params := paramtable.Get()
	for _, item := range []*paramtable.ParamItem{&params.ProxyCfg.EnablePredicateOptimization, &params.CommonCfg.EnabledOptimizeExpr} {
		key, previous := item.Key, item.GetValue()
		require.NoError(b, params.Save(key, "false"))
		b.Cleanup(func() { require.NoError(b, params.Save(key, previous)) })
	}
	helper, err := typeutil.CreateSchemaHelper(touchstone.Schema("benchmark", false))
	require.NoError(b, err)
	view, err := schema.NewView(helper, helper.TopLevelFields())
	require.NoError(b, err)
	for _, workload := range touchstone.PerformanceWorkloads() {
		b.Run(workload.Name, func(b *testing.B) {
			require.NoError(b, params.Save(params.ProxyCfg.EnablePredicateOptimization.Key, "false"))
			original, err := planparserv2.ParseExpr(helper, workload.Query, nil)
			require.NoError(b, err)
			input := orchestration.AttemptInput{Original: original, SchemaView: view}
			for _, mode := range []struct {
				name string
				run  func(context.Context, orchestration.AttemptInput, func(*planpb.Expr) error) (orchestration.AttemptResult, error)
			}{{"empty", orchestration.RoundTrip}, {"five_passes", orchestration.Optimize}} {
				b.Run(mode.name, func(b *testing.B) {
					ctx, validate := context.Background(), func(*planpb.Expr) error { return nil }
					result, err := mode.run(ctx, input, validate)
					require.NoError(b, err)
					require.Equal(b, orchestration.OutcomeTranslated, result.Outcome())
					outputSize := proto.Size(result.Candidate())
					b.ReportAllocs()
					b.ResetTimer()
					for i := 0; i < b.N; i++ {
						result, err = mode.run(ctx, input, validate)
						if err != nil || result.Outcome() != orchestration.OutcomeTranslated {
							b.Fatalf("pipeline failed: %v %v", result.Report(), err)
						}
					}
					b.StopTimer()
					b.ReportMetric(float64(proto.Size(original)), "input-B")
					b.ReportMetric(float64(outputSize), "output-B")
				})
			}
			for _, enabled := range []bool{false, true} {
				b.Run("plan_enabled="+strconv.FormatBool(enabled), func(b *testing.B) {
					require.NoError(b, params.Save(params.ProxyCfg.EnablePredicateOptimization.Key, strconv.FormatBool(enabled)))
					plan, err := planparserv2.CreateRetrievePlan(helper, workload.Query, nil)
					require.NoError(b, err)
					b.ReportAllocs()
					b.ResetTimer()
					for i := 0; i < b.N; i++ {
						plan, err = planparserv2.CreateRetrievePlan(helper, workload.Query, nil)
						if err != nil {
							b.Fatal(err)
						}
					}
					b.StopTimer()
					b.ReportMetric(float64(proto.Size(plan)), "plan-B")
				})
			}
		})
	}
}

// Records complete parser-built plan sizes under both legacy-optimizer settings.
func TestPredicatePlanSizes(t *testing.T) {
	paramtable.Init()
	params := paramtable.Get()
	for _, item := range []*paramtable.ParamItem{&params.ProxyCfg.EnablePredicateOptimization, &params.CommonCfg.EnabledOptimizeExpr} {
		key, previous := item.Key, item.GetValue()
		t.Cleanup(func() { require.NoError(t, params.Save(key, previous)) })
	}
	helper, err := typeutil.CreateSchemaHelper(touchstone.Schema("sizes", false))
	require.NoError(t, err)
	for _, legacy := range []bool{false, true} {
		require.NoError(t, params.Save(params.CommonCfg.EnabledOptimizeExpr.Key, strconv.FormatBool(legacy)))
		for _, workload := range touchstone.PerformanceWorkloads() {
			for _, enabled := range []bool{false, true} {
				require.NoError(t, params.Save(params.ProxyCfg.EnablePredicateOptimization.Key, strconv.FormatBool(enabled)))
				plan, err := planparserv2.CreateRetrievePlan(helper, workload.Query, nil)
				require.NoError(t, err)
				data, err := proto.Marshal(plan)
				require.NoError(t, err)
				t.Logf("PLAN_SIZE workload=%s legacy=%t enabled=%t expr_bytes=%d plan_bytes=%d", workload.Name, legacy, enabled, proto.Size(plan.GetQuery().GetPredicates()), len(data))
			}
		}
	}
}

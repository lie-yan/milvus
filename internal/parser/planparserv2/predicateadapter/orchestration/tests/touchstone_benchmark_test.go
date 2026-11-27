package orchestration_test

import (
	"context"
	"testing"

	"github.com/stretchr/testify/require"

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

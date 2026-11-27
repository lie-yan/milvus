package translation_test

import (
	"context"
	"slices"
	"strconv"
	"testing"

	"github.com/lie-yan/predicate-optimization/concrete_rewriters/array_exclusion_grouping_rewriter"
	"github.com/lie-yan/predicate-optimization/concrete_rewriters/array_membership_normalization_rewriter"
	"github.com/lie-yan/predicate-optimization/concrete_rewriters/boolean_cleanup_rewriter"
	"github.com/lie-yan/predicate-optimization/concrete_rewriters/logical_flattening_rewriter"
	"github.com/lie-yan/predicate-optimization/concrete_rewriters/scoped_contextual_rewriter"
	"github.com/lie-yan/predicate-optimization/expressions"
	"github.com/lie-yan/predicate-optimization/rewriter"
	"github.com/stretchr/testify/require"
	"google.golang.org/protobuf/proto"

	"github.com/milvus-io/milvus/internal/parser/planparserv2"
	adapterruntime "github.com/milvus-io/milvus/internal/parser/planparserv2/predicateadapter/runtime"
	"github.com/milvus-io/milvus/internal/parser/planparserv2/predicateadapter/schema"
	"github.com/milvus-io/milvus/internal/parser/planparserv2/predicateadapter/tests/touchstone"
	"github.com/milvus-io/milvus/internal/parser/planparserv2/predicateadapter/translation"
	"github.com/milvus-io/milvus/pkg/v3/proto/planpb"
	"github.com/milvus-io/milvus/pkg/v3/util/paramtable"
	"github.com/milvus-io/milvus/pkg/v3/util/typeutil"
)

// Exercises real pass outputs without enabling any pass in production orchestration.
func TestTouchstoneRewrittenExport(t *testing.T) {
	paramtable.Init()
	params := paramtable.Get()
	for _, item := range []*paramtable.ParamItem{&params.ProxyCfg.EnablePredicateOptimization, &params.CommonCfg.EnabledOptimizeExpr} {
		previous, key := item.GetValue(), item.Key
		t.Cleanup(func() { require.NoError(t, params.Save(key, previous)) })
	}
	require.NoError(t, params.Save(params.ProxyCfg.EnablePredicateOptimization.Key, "false"))
	runtime, err := adapterruntime.SharedRuntime()
	require.NoError(t, err)
	provider := &noEvaluationContext{runtime.FunctionContext()}
	for _, nullable := range []bool{false, true} {
		for _, legacy := range []string{"false", "true"} {
			t.Run(strconv.FormatBool(nullable)+"/legacy="+legacy, func(t *testing.T) {
				require.NoError(t, params.Save(params.ProxyCfg.EnablePredicateOptimization.Key, "false"))
				require.NoError(t, params.Save(params.CommonCfg.EnabledOptimizeExpr.Key, legacy))
				collection := touchstone.Schema("touchstone", nullable)
				helper, err := typeutil.CreateSchemaHelper(collection)
				require.NoError(t, err)
				view, err := schema.NewView(helper, collection.Fields)
				require.NoError(t, err)
				translator, err := translation.New(&guardedRuntime{Runtime: runtime, provider: provider}, view, runtime.Types())
				require.NoError(t, err)
				source, err := planparserv2.ParseExpr(helper, touchstone.Query, nil)
				require.NoError(t, err)
				before := proto.Clone(source)
				imported := importSource(t, translator, source)
				cases, err := touchstone.Cases(nullable)
				require.NoError(t, err)
				graph := imported.Root()
				bindings := map[expressions.Expression]bool{}
				visitTouchstoneGraph(graph, func(node expressions.Expression) {
					switch node.(type) {
					case *expressions.ReferenceExpression, *expressions.OpaqueExpression:
						bindings[node] = true
					}
				})
				for _, pass := range []struct {
					name string
					rule rewriter.ExpressionRewriter[struct{}]
				}{
					{"contextual", scoped_contextual_rewriter.NewScopedContextualRewriter(runtime.ValueFactory(), provider)},
					{"cleanup", boolean_cleanup_rewriter.NewBooleanCleanupRewriter(runtime.ValueFactory(), provider)},
					{"flatten", logical_flattening_rewriter.NewLogicalFlatteningRewriter()},
					{"normalize", array_membership_normalization_rewriter.NewArrayMembershipNormalizationRewriter()},
					{"group", array_exclusion_grouping_rewriter.NewArrayExclusionGroupingRewriter(runtime.ValueFactory())},
				} {
					graph, err = rewriter.NewExpressionTreeRewriter[struct{}](pass.rule).Rewrite(graph, struct{}{})
					require.NoError(t, err, pass.name)
					require.True(t, imported.Root().Type().EqualsTo(graph.Type()), pass.name)
					output, err := translator.Export(context.Background(), imported, graph)
					require.NoError(t, err, pass.name)
					for _, tc := range cases {
						row := touchstoneRow(tc.Row)
						want := map[string]int{"UNKNOWN": -1, "FALSE": 0, "TRUE": 1}[tc.Expected]
						require.Equal(t, want, nullablePredicate(t, source, row), tc.Name)
						require.Equal(t, want, nullablePredicate(t, output, row), pass.name+"/"+tc.Name)
					}
					require.True(t, proto.Equal(before, source), pass.name)
					nodes := 0
					visitTouchstoneGraph(graph, func(node expressions.Expression) {
						nodes++
						switch node.(type) {
						case *expressions.ReferenceExpression, *expressions.OpaqueExpression:
							require.True(t, bindings[node], "rewritten leaves must retain session bindings")
						}
					})
					t.Logf("%s: %d node occurrences", pass.name, nodes)
					if pass.name == "group" {
						assertTouchstoneGroupedTags(t, output, nullable)
						// Re-exporting remains independent of mutations to a previous result.
						snapshot := proto.Clone(output)
						output.Reset()
						again, err := translator.Export(context.Background(), imported, graph)
						require.NoError(t, err)
						require.True(t, proto.Equal(snapshot, again))
					}
				}
				// Exercise the actual parser hook and composed production pipeline as well.
				require.NoError(t, params.Save(params.ProxyCfg.EnablePredicateOptimization.Key, "true"))
				wired, err := planparserv2.ParseExpr(helper, touchstone.Query, nil)
				require.NoError(t, err)
				require.False(t, proto.Equal(source, wired), "production hook must apply the rewriters")
				assertTouchstoneGroupedTags(t, wired, nullable)
				for _, tc := range cases {
					want := map[string]int{"UNKNOWN": -1, "FALSE": 0, "TRUE": 1}[tc.Expected]
					require.Equal(t, want, nullablePredicate(t, wired, touchstoneRow(tc.Row)), tc.Name)
				}
			})
		}
	}
}

// Converts the independent dataset to oracle payloads using the fixture's fixed field IDs.
func touchstoneRow(row touchstone.Row) map[int64]*planpb.GenericValue {
	values := map[int64]*planpb.GenericValue{101: nil, 102: nil, 103: nil, 104: nil, 105: nil, 106: nil}
	if row.Article != nil {
		values[101] = stringSource(*row.Article)
	}
	if row.Review != nil {
		values[102] = integerSource(*row.Review)
	}
	if row.Brand != nil {
		values[103] = integerSource(*row.Brand)
	}
	for i, items := range [][]int64{row.Tags, row.Categories, row.Malls} {
		if items == nil {
			continue
		}
		array := &planpb.Array{}
		for _, item := range items {
			array.Array = append(array.Array, integerSource(item))
		}
		values[int64(104+i)] = &planpb.GenericValue{Val: &planpb.GenericValue_ArrayVal{ArrayVal: array}}
	}
	return values
}

// Traverses occurrences so shared children are checked in every output position.
func visitTouchstoneGraph(node expressions.Expression, visit func(expressions.Expression)) {
	visit(node)
	for i := 0; i < node.ChildCount(); i++ {
		visitTouchstoneGraph(node.ChildAt(i), visit)
	}
}

// Distinguishes real grouping from an unchanged but semantically correct export.
func assertTouchstoneGroupedTags(t *testing.T, node *planpb.Expr, nullable bool) {
	t.Helper()
	var grouped [][]int64
	var visit func(*planpb.Expr)
	visit = func(expr *planpb.Expr) {
		if binary := expr.GetBinaryExpr(); binary != nil {
			visit(binary.Left)
			visit(binary.Right)
		} else if unary := expr.GetUnaryExpr(); unary != nil {
			visit(unary.Child)
		} else if contains := expr.GetJsonContainsExpr(); contains != nil {
			require.True(t, contains.ElementsSameType)
			require.True(t, contains.ColumnInfo.Nullable)
			if contains.ColumnInfo.FieldId == 105 || contains.ColumnInfo.FieldId == 106 {
				require.Equal(t, planpb.JSONContainsExpr_Contains, contains.Op, "singleton membership must normalize")
			}
			if contains.ColumnInfo.FieldId == 104 && contains.Op == planpb.JSONContainsExpr_ContainsAny {
				var candidates []int64
				for _, candidate := range contains.Elements {
					candidates = append(candidates, candidate.GetInt64Val())
				}
				slices.Sort(candidates)
				grouped = append(grouped, candidates)
			}
		}
	}
	visit(node)
	expected := [][]int64{{8791003, 8791004, 8791005, 8791006, 8791007, 8791008, 8791009}}
	if nullable {
		expected = [][]int64{{8791001, 8791002}, {8791003, 8791006, 8791009}}
	}
	require.ElementsMatch(t, expected, grouped)
}

package planparserv2

import (
	"context"
	"strconv"
	"testing"

	"github.com/bytedance/mockey"
	"github.com/lie-yan/predicate-optimization/expressions"
	shared "github.com/lie-yan/predicate-optimization/functions"
	functioninterfaces "github.com/lie-yan/predicate-optimization/functions/interfaces"
	valueinterfaces "github.com/lie-yan/predicate-optimization/values/interfaces"
	"github.com/stretchr/testify/require"
	"google.golang.org/protobuf/proto"

	"github.com/milvus-io/milvus-proto/go-api/v3/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
	adapterfunctions "github.com/milvus-io/milvus/internal/parser/planparserv2/predicateadapter/functions"
	"github.com/milvus-io/milvus/internal/parser/planparserv2/predicateadapter/integration"
	"github.com/milvus-io/milvus/internal/parser/planparserv2/predicateadapter/orchestration"
	adapterruntime "github.com/milvus-io/milvus/internal/parser/planparserv2/predicateadapter/runtime"
	adapterschema "github.com/milvus-io/milvus/internal/parser/planparserv2/predicateadapter/schema"
	"github.com/milvus-io/milvus/internal/parser/planparserv2/predicateadapter/tests/touchstone"
	"github.com/milvus-io/milvus/internal/parser/planparserv2/predicateadapter/translation"
	"github.com/milvus-io/milvus/pkg/v3/proto/planpb"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
	"github.com/milvus-io/milvus/pkg/v3/util/paramtable"
	"github.com/milvus-io/milvus/pkg/v3/util/typeutil"
)

func enablePredicateOptimization(t *testing.T) {
	t.Helper()
	params := paramtable.Get()
	item := &params.ProxyCfg.EnablePredicateOptimization
	previous := item.GetValue()
	require.NoError(t, params.Save(item.Key, "true"))
	t.Cleanup(func() { require.NoError(t, params.Save(item.Key, previous)) })
}

func TestPredicateOptimizationPostLegacyHook(t *testing.T) {
	enablePredicateOptimization(t)
	schema := newTestSchema(true)
	for _, field := range schema.Fields {
		if field.DataType == schemapb.DataType_VarChar {
			field.TypeParams = []*commonpb.KeyValuePair{{Key: "max_length", Value: "16"}}
		}
	}
	schema.Fields = append(schema.Fields, &schemapb.FieldSchema{FieldID: 2000, Name: "NullableInt", DataType: schemapb.DataType_Int64, Nullable: true})
	helper, err := typeutil.CreateSchemaHelper(schema)
	require.NoError(t, err)
	evaluations := mockey.Mock((*adapterfunctions.FunctionContext).Evaluate).To(func(*adapterfunctions.FunctionContext, functioninterfaces.FunctionHandle, *shared.FunctionSignature, []valueinterfaces.Value) (valueinterfaces.Value, error) {
		t.Fatal("empty pipeline evaluated a function")
		return nil, merr.ErrServiceInternal
	}).Build()
	defer evaluations.UnPatch()
	for _, source := range []string{
		"Int64Field in [7]", "true", "false", "true or Int64Field > 7",
		"BoolField != false", "3 <= Int64Field < 12", "Int64Field < FieldID",
		"Int64Field in [9, 2, 9, 0]", "Int64Field > 7 and NullableInt is null",
		"3 <= Int32Field < 12", "3.0 <= FloatField < 12.0", `VarCharField in ["0", "2"]`,
	} {
		t.Run(source, func(t *testing.T) {
			params := paramtable.Get()
			key := params.ProxyCfg.EnablePredicateOptimization.Key
			require.NoError(t, params.Save(key, "false"))
			legacy, err := ParseExpr(helper, source, nil)
			require.NoError(t, err)
			require.NoError(t, params.Save(key, "true"))
			observations, attempts := 0, 0
			var borrowed, before *planpb.Expr
			var run func(context.Context, orchestration.AttemptInput, func(*planpb.Expr) error) (orchestration.AttemptResult, error)
			patch := mockey.Mock(orchestration.RoundTrip).Origin(&run).To(func(ctx context.Context, input orchestration.AttemptInput, validate func(*planpb.Expr) error) (orchestration.AttemptResult, error) {
				attempts++
				require.Equal(t, 1, observations)
				require.Nil(t, ctx.Done(), "production translation must run to completion")
				require.True(t, proto.Equal(legacy, input.Original), "hook must see legacy output")
				borrowed, before = input.Original, proto.Clone(input.Original).(*planpb.Expr)
				result, err := run(ctx, input, validate)
				require.NoError(t, err)
				require.Equal(t, orchestration.OutcomeTranslated, result.Outcome())
				return result, err
			}).Build()
			defer patch.UnPatch()
			got, err := parseExprInner(helper, source, nil, &ParserVisitorArgs{OnParsedExpr: func(*planpb.Expr) { observations++ }})
			require.NoError(t, err)
			require.Equal(t, 1, attempts)
			require.True(t, proto.Equal(legacy, got))
			require.NotSame(t, borrowed, got)
			require.True(t, proto.Equal(before, borrowed))
		})
	}
}

func TestPredicateOptimizationBypassAndOriginalErrors(t *testing.T) {
	enablePredicateOptimization(t)
	helper := newTestSchemaHelper(t)
	views := mockey.Mock(adapterschema.NewView).To(func(*typeutil.SchemaHelper, []*schemapb.FieldSchema) (*adapterschema.View, error) {
		t.Fatal("bypassed parse constructed a view")
		return nil, merr.ErrServiceInternal
	}).Build()
	defer views.UnPatch()
	runtime := mockey.Mock(adapterruntime.SharedRuntime).To(func() (*adapterruntime.Runtime, error) {
		t.Fatal("bypassed parse acquired runtime")
		return nil, merr.ErrServiceInternal
	}).Build()
	defer runtime.UnPatch()
	// The disabled path performs no adapter setup, even for supported predicates.
	params := paramtable.Get()
	key := params.ProxyCfg.EnablePredicateOptimization.Key
	require.NoError(t, params.Save(key, "false"))
	_, err := ParseExpr(helper, "Int64Field > 7", nil)
	require.NoError(t, err)
	require.NoError(t, params.Save(key, "true"))
	for _, source := range []string{"", " \t\n"} {
		got, err := parseExprInner(helper, source, nil, nil)
		require.NoError(t, err)
		require.NotNil(t, got.GetAlwaysTrueExpr())
	}
	_, err = parseExprInner(helper, "Int64Field >", nil, nil)
	require.Error(t, err)
	require.False(t, integration.IsFailure(err))
	placement := mockey.Mock(validateElementFilterPlacement).Return(merr.ErrQueryPlan).Build()
	defer placement.UnPatch()
	_, err = parseExprInner(helper, "Int64Field > 7", nil, nil)
	require.ErrorIs(t, err, merr.ErrQueryPlan)
	require.False(t, integration.IsFailure(err))
}

func TestPredicateOptimizationMembershipAccounting(t *testing.T) {
	enablePredicateOptimization(t)
	helper := newTestSchemaHelper(t)
	template, _ := roaringBytesTemplate(t, 1, 2, 3)
	values := map[string]*schemapb.TemplateValue{"rb": template}
	const source = "membership_match(Int64Field, {rb}, type=roaring)"
	legacyBudget, enabledBudget := NewMembershipPreflightBudget(), NewMembershipPreflightBudget()
	params := paramtable.Get()
	key := params.ProxyCfg.EnablePredicateOptimization.Key
	require.NoError(t, params.Save(key, "false"))
	legacy, err := parseExprInner(helper, source, values, &ParserVisitorArgs{MembershipBudget: legacyBudget})
	require.NoError(t, err)
	require.NoError(t, params.Save(key, "true"))
	observations := 0
	got, err := parseExprInner(helper, source, values, &ParserVisitorArgs{
		MembershipBudget: enabledBudget,
		OnParsedExpr:     func(*planpb.Expr) { observations++ },
	})
	require.NoError(t, err)
	require.Equal(t, 1, observations)
	require.True(t, proto.Equal(legacy, got))
	require.Equal(t, legacyBudget.occurrenceBytes, enabledBudget.occurrenceBytes)
	require.Equal(t, legacyBudget.aggregateDecodedBytes, enabledBudget.aggregateDecodedBytes)
	require.Equal(t, len(legacyBudget.validated), len(enabledBudget.validated))
}

func TestPredicateOptimizationRuntimeSwitch(t *testing.T) {
	enablePredicateOptimization(t)
	helper := newTestSchemaHelper(t)
	params := paramtable.Get()
	key := params.ProxyCfg.EnablePredicateOptimization.Key
	attempts := 0
	patch := mockey.Mock(integration.Apply).To(func(_ *typeutil.SchemaHelper, original *planpb.Expr, _ func(*planpb.Expr) error) (*planpb.Expr, error) {
		attempts++
		return proto.Clone(original).(*planpb.Expr), nil
	}).Build()
	defer patch.UnPatch()
	// Change the switch during parsing: only the following parse sees the refresh.
	for _, tc := range []struct {
		initial, updated string
		want             int
	}{
		{"false", "true", 0}, {"true", "false", 1},
	} {
		require.NoError(t, params.Save(key, tc.initial))
		attempts = 0
		_, err := parseExprInner(helper, "Int64Field > 7", nil, &ParserVisitorArgs{
			OnParsedExpr: func(*planpb.Expr) { require.NoError(t, params.Save(key, tc.updated)) },
		})
		require.NoError(t, err)
		require.Equal(t, tc.want, attempts)
		attempts = 0
		_, err = parseExprInner(helper, "Int64Field > 7", nil, nil)
		require.NoError(t, err)
		require.Equal(t, 1-tc.want, attempts)
	}
	require.NoError(t, params.Save(key, "true"))
	attempts = 0
	_, err := ParseExpr(helper, "Int64Field > 7", nil)
	require.NoError(t, err)
	require.Equal(t, 1, attempts, "the runtime switch must control the public parse path")
}

func TestPredicateOptimizationScorerSwitchCapture(t *testing.T) {
	enablePredicateOptimization(t)
	helper := newTestSchemaHelper(t)
	params := paramtable.Get()
	key := params.ProxyCfg.EnablePredicateOptimization.Key
	for _, initial := range []bool{false, true} {
		t.Run(strconv.FormatBool(initial), func(t *testing.T) {
			require.NoError(t, params.Save(key, strconv.FormatBool(initial)))
			observed := 0
			var translatedValues []int64
			var run func(context.Context, orchestration.AttemptInput, func(*planpb.Expr) error) (orchestration.AttemptResult, error)
			patch := mockey.Mock(orchestration.RoundTrip).Origin(&run).To(func(ctx context.Context, in orchestration.AttemptInput, validate func(*planpb.Expr) error) (orchestration.AttemptResult, error) {
				result, err := run(ctx, in, validate)
				require.NoError(t, err)
				require.Equal(t, orchestration.OutcomeTranslated, result.Outcome())
				translatedValues = append(translatedValues, in.Original.GetUnaryRangeExpr().GetValue().GetInt64Val())
				return result, err
			}).Build()
			defer patch.UnPatch()
			score := &schemapb.FunctionScore{Functions: []*schemapb.FunctionSchema{{Type: schemapb.FunctionType_Rerank, Params: []*commonpb.KeyValuePair{{Key: "reranker", Value: "boost"}, {Key: "filter", Value: "Int64Field < 20"}, {Key: "weight", Value: "2"}}}}}
			plan, err := CreateSearchPlanArgs(helper, "Int64Field > 7", "FloatVectorField", &planpb.QueryInfo{GroupByFieldId: -1}, nil, score, &ParserVisitorArgs{OnParsedExpr: func(*planpb.Expr) {
				observed++
				if observed == 1 {
					require.NoError(t, params.Save(key, strconv.FormatBool(!initial)))
				}
			}})
			require.NoError(t, err)
			require.Len(t, plan.GetScorers(), 1)
			require.Equal(t, 2, observed)
			expected := int64(20)
			if initial {
				expected = 7
			}
			require.Equal(t, []int64{expected}, translatedValues, "main keeps its entry snapshot; scorer captures the refreshed value")
		})
	}
}

func TestPredicateOptimizationScorerMembershipBudget(t *testing.T) {
	enablePredicateOptimization(t)
	helper := newTestSchemaHelper(t)
	template, _ := roaringBytesTemplate(t, 1, 2, 3)
	values := map[string]*schemapb.TemplateValue{"rb": template}
	const filter = "membership_match(Int64Field, {rb}, type=roaring)"
	score := &schemapb.FunctionScore{Functions: []*schemapb.FunctionSchema{{Type: schemapb.FunctionType_Rerank, Params: []*commonpb.KeyValuePair{{Key: "reranker", Value: "boost"}, {Key: "filter", Value: filter}, {Key: "weight", Value: "2"}}}}}
	single := NewMembershipPreflightBudget()
	_, err := parseExprInner(helper, filter, values, &ParserVisitorArgs{MembershipBudget: single})
	require.NoError(t, err)
	require.Positive(t, single.occurrenceBytes)
	params := paramtable.Get()
	limit := &params.ProxyCfg.MaxMembershipFilterPlanSize
	previous := limit.GetValue()
	t.Cleanup(func() { require.NoError(t, params.Save(limit.Key, previous)) })
	for _, enabled := range []string{"false", "true"} {
		t.Run(enabled, func(t *testing.T) {
			require.NoError(t, params.Save(params.ProxyCfg.EnablePredicateOptimization.Key, enabled))
			require.NoError(t, params.Save(limit.Key, previous))
			budget := NewMembershipPreflightBudget()
			_, err := CreateSearchPlanArgs(helper, filter, "FloatVectorField", &planpb.QueryInfo{GroupByFieldId: -1}, values, score, &ParserVisitorArgs{MembershipBudget: budget})
			require.NoError(t, err)
			require.Equal(t, 2*single.occurrenceBytes, budget.occurrenceBytes)
			require.Equal(t, 2*single.aggregateDecodedBytes, budget.aggregateDecodedBytes, "decoded memory is charged per occurrence despite shared validation")
			require.Len(t, budget.validated, len(single.validated))
			// Both serialized and decoded sizes use this limit; one occurrence must fit.
			combinedLimit := max(2*single.occurrenceBytes, int64(2*single.aggregateDecodedBytes)) - 1
			require.NoError(t, params.Save(limit.Key, strconv.FormatInt(combinedLimit, 10)))
			_, err = CreateSearchPlanArgs(helper, filter, "FloatVectorField", &planpb.QueryInfo{GroupByFieldId: -1}, values, nil, nil)
			require.NoError(t, err, "main predicate alone must fit before testing the shared scorer budget")
			_, err = CreateSearchPlanArgs(helper, filter, "FloatVectorField", &planpb.QueryInfo{GroupByFieldId: -1}, values, score, nil)
			require.ErrorIs(t, err, merr.ErrParameterTooLarge, "scorer must consume the main predicate's occurrence budget")
		})
	}
}

// Exercises real parsing, template binding and both legacy rewrite modes before the adapter.
func TestPredicateOptimizationNullableScalarWiring(t *testing.T) {
	enablePredicateOptimization(t)
	params := paramtable.Get()
	legacyKey := params.CommonCfg.EnabledOptimizeExpr.Key
	previous := params.CommonCfg.EnabledOptimizeExpr.GetValue()
	t.Cleanup(func() { require.NoError(t, params.Save(legacyKey, previous)) })
	evaluations := mockey.Mock((*adapterfunctions.FunctionContext).Evaluate).To(func(*adapterfunctions.FunctionContext, functioninterfaces.FunctionHandle, *shared.FunctionSignature, []valueinterfaces.Value) (valueinterfaces.Value, error) {
		t.Fatal("empty scalar pipeline evaluated a function")
		return nil, merr.ErrServiceInternal
	}).Build()
	defer evaluations.UnPatch()
	for _, nullable := range []bool{false, true} {
		fields := []*schemapb.FieldSchema{
			{FieldID: 101, Name: "article_type", DataType: schemapb.DataType_VarChar, Nullable: nullable, TypeParams: []*commonpb.KeyValuePair{{Key: "max_length", Value: "32"}}},
			{FieldID: 102, Name: "is_review", DataType: schemapb.DataType_Int64, Nullable: true},
			{FieldID: 103, Name: "brand_id", DataType: schemapb.DataType_Int64, Nullable: true},
		}
		helper, err := typeutil.CreateSchemaHelper(&schemapb.CollectionSchema{Fields: fields})
		require.NoError(t, err)
		for _, legacyEnabled := range []string{"false", "true"} {
			require.NoError(t, params.Save(legacyKey, legacyEnabled))
			for _, query := range []struct {
				source, expanded string
			}{
				{`article_type == "tsa_04"`, ""},
				{`article_type != "tsa_04"`, ""},
				{`article_type in ["tsa_03", "tsa_01", "tsa_03"] and is_review in [8795001,8795002] and brand_id not in [8798001,8798002]`, ""},
				{`not (article_type in ["tsa_01"] or is_review not in [8795001, 8795002])`, ""},
				{`is_review in [8795001] and article_type == "tsa_04"`, ""},
				{`article_type == {text}`, `article_type == "猫"`},
			} {
				t.Run(strconv.FormatBool(nullable)+"/"+legacyEnabled+"/"+query.source, func(t *testing.T) {
					key := params.ProxyCfg.EnablePredicateOptimization.Key
					require.NoError(t, params.Save(key, "false"))
					template := map[string]*schemapb.TemplateValue{"text": {Val: &schemapb.TemplateValue_StringVal{StringVal: "猫"}}}
					legacy, err := ParseExpr(helper, query.source, template)
					require.NoError(t, err)
					expected := legacy
					if query.expanded != "" {
						expected, err = ParseExpr(helper, query.expanded, nil)
						require.NoError(t, err)
					}
					require.NoError(t, params.Save(key, "true"))
					attempts, imports := 0, 0
					var originalImport func(*translation.Translator, context.Context, *planpb.Expr) (*translation.ImportResult, error)
					inspect := mockey.Mock((*translation.Translator).Import).Origin(&originalImport).To(func(translator *translation.Translator, ctx context.Context, source *planpb.Expr) (*translation.ImportResult, error) {
						imports++
						result, err := originalImport(translator, ctx, source)

						require.NoError(t, err)
						var walk func(expressions.Expression)
						walk = func(node expressions.Expression) {
							_, opaque := node.(*expressions.OpaqueExpression)
							require.False(t, opaque, "scalar coverage must be exposed")
							for i := 0; i < node.ChildCount(); i++ {
								walk(node.ChildAt(i))
							}
						}
						walk(result.Root())
						return result, err
					}).Build()
					defer inspect.UnPatch()
					var run func(context.Context, orchestration.AttemptInput, func(*planpb.Expr) error) (orchestration.AttemptResult, error)
					patch := mockey.Mock(orchestration.RoundTrip).Origin(&run).To(func(ctx context.Context, input orchestration.AttemptInput, validate func(*planpb.Expr) error) (orchestration.AttemptResult, error) {
						attempts++
						require.True(t, proto.Equal(legacy, input.Original))
						before := proto.Clone(input.Original)
						result, err := run(ctx, input, validate)
						require.NoError(t, err)
						require.Equal(t, orchestration.OutcomeTranslated, result.Outcome())
						require.True(t, proto.Equal(before, input.Original))
						return result, err
					}).Build()
					defer patch.UnPatch()
					actual, err := ParseExpr(helper, query.source, template)
					require.NoError(t, err)
					require.Equal(t, 1, attempts)
					require.Equal(t, 1, imports)
					require.True(t, proto.Equal(expected, actual), "export differs from bound legacy predicate: %s", actual)
				})
			}
		}
	}
}

// Requires actual parser-produced scalar plans to translate across widths and nullability.
func TestPredicateOptimizationScalarDomains(t *testing.T) {
	enablePredicateOptimization(t)
	var current *testing.T
	attempts := 0
	var run func(context.Context, orchestration.AttemptInput, func(*planpb.Expr) error) (orchestration.AttemptResult, error)
	patch := mockey.Mock(orchestration.RoundTrip).Origin(&run).To(func(ctx context.Context, input orchestration.AttemptInput, validate func(*planpb.Expr) error) (orchestration.AttemptResult, error) {
		attempts++
		result, err := run(ctx, input, validate)
		require.NoError(current, err)
		require.Equal(current, orchestration.OutcomeTranslated, result.Outcome())
		return result, err
	}).Build()
	defer patch.UnPatch()
	for _, tc := range []struct {
		typ     schemapb.DataType
		literal string
	}{
		{schemapb.DataType_Bool, "true"},
		{schemapb.DataType_Int8, "127"},
		{schemapb.DataType_Int16, "32767"},
		{schemapb.DataType_Int32, "2147483647"},
		{schemapb.DataType_Int64, "9007199254740993"},
		{schemapb.DataType_Float, "1.5"},
		{schemapb.DataType_Double, "1.1"},
		{schemapb.DataType_VarChar, `"猫"`},
	} {
		for _, nullable := range []bool{false, true} {
			fields := []*schemapb.FieldSchema{
				{FieldID: 101, Name: "x", DataType: tc.typ, Nullable: nullable},
				{FieldID: 102, Name: "y", DataType: tc.typ},
			}
			if tc.typ == schemapb.DataType_VarChar {
				for _, field := range fields {
					field.TypeParams = []*commonpb.KeyValuePair{{Key: "max_length", Value: "32"}}
				}
			}
			helper, err := typeutil.CreateSchemaHelper(&schemapb.CollectionSchema{Fields: fields})
			require.NoError(t, err)
			queries := []string{"x == " + tc.literal, "x == y", "x in [" + tc.literal + "]"}
			if tc.typ != schemapb.DataType_Bool && tc.typ != schemapb.DataType_VarChar {
				queries = append(queries, "0.0 < x < 2.0")
				if tc.typ != schemapb.DataType_Float && tc.typ != schemapb.DataType_Double {
					queries[len(queries)-1] = "0 < x < 2"
				}
			}
			for _, query := range queries {
				t.Run(tc.typ.String()+"/"+strconv.FormatBool(nullable)+"/"+query, func(t *testing.T) {
					current, attempts = t, 0
					_, err := ParseExpr(helper, query, nil)
					require.NoError(t, err)
					require.Equal(t, 1, attempts)
				})
			}
		}
	}
}

func TestPredicateOptimizationArrayWiring(t *testing.T) {
	enablePredicateOptimization(t)
	var current *testing.T
	attempts := 0
	var run func(context.Context, orchestration.AttemptInput, func(*planpb.Expr) error) (orchestration.AttemptResult, error)
	patch := mockey.Mock(orchestration.RoundTrip).Origin(&run).To(func(ctx context.Context, input orchestration.AttemptInput, validate func(*planpb.Expr) error) (orchestration.AttemptResult, error) {
		attempts++
		before := proto.Clone(input.Original)
		result, err := run(ctx, input, validate)
		require.NoError(current, err)
		require.Equal(current, orchestration.OutcomeTranslated, result.Outcome())
		require.True(current, proto.Equal(before, input.Original))
		return result, err
	}).Build()
	defer patch.UnPatch()
	evaluations := mockey.Mock((*adapterfunctions.FunctionContext).Evaluate).To(func(*adapterfunctions.FunctionContext, functioninterfaces.FunctionHandle, *shared.FunctionSignature, []valueinterfaces.Value) (valueinterfaces.Value, error) {
		t.Fatal("empty array pipeline evaluated a function")
		return nil, merr.ErrServiceInternal
	}).Build()
	defer evaluations.UnPatch()
	for _, tc := range []struct {
		element schemapb.DataType
		literal string
	}{
		{schemapb.DataType_Bool, "true"},
		{schemapb.DataType_Int8, "127"},
		{schemapb.DataType_Int16, "32767"},
		{schemapb.DataType_Int32, "2147483647"},
		{schemapb.DataType_Int64, "9007199254740993"},
		{schemapb.DataType_Float, "1.5"},
		{schemapb.DataType_Double, "1.1"},
	} {
		for _, nullable := range []bool{false, true} {
			helper, err := typeutil.CreateSchemaHelper(&schemapb.CollectionSchema{Fields: []*schemapb.FieldSchema{
				{FieldID: 101, Name: "items", DataType: schemapb.DataType_Array, ElementType: tc.element, Nullable: nullable, TypeParams: []*commonpb.KeyValuePair{{Key: "max_capacity", Value: "32"}}},
			}})
			require.NoError(t, err)
			queries := []string{
				"array_contains(items, " + tc.literal + ")",
				"not array_contains_any(items, [" + tc.literal + "," + tc.literal + "])",
				"array_contains_any(items, [])",
				"array_contains(items, " + tc.literal + ") and not array_contains_any(items, [])",
			}
			if tc.element == schemapb.DataType_Int64 {
				queries = append(queries, "array_contains(items, {candidate})", "array_contains_any(items, {candidates})")
			}
			for _, query := range queries {
				t.Run(tc.element.String()+"/"+strconv.FormatBool(nullable)+"/"+query, func(t *testing.T) {
					current, attempts = t, 0
					output, err := ParseExpr(helper, query, map[string]*schemapb.TemplateValue{
						"candidate":  {Val: &schemapb.TemplateValue_Int64Val{Int64Val: 7}},
						"candidates": {Val: &schemapb.TemplateValue_ArrayVal{ArrayVal: &schemapb.TemplateArrayValue{Data: &schemapb.TemplateArrayValue_LongData{LongData: &schemapb.LongArray{Data: []int64{7, 9}}}}}},
					})
					require.NoError(t, err)
					require.Equal(t, 1, attempts)
					if node := output.GetJsonContainsExpr(); node != nil {
						require.Empty(t, node.TemplateVariableName)
						require.False(t, output.IsTemplate)
					}
				})
			}
		}
	}
}

func TestPredicateOptimizationTouchstoneWiring(t *testing.T) {
	enablePredicateOptimization(t)
	params := paramtable.Get()
	legacyKey := params.CommonCfg.EnabledOptimizeExpr.Key
	previous := params.CommonCfg.EnabledOptimizeExpr.GetValue()
	t.Cleanup(func() { require.NoError(t, params.Save(legacyKey, previous)) })
	evaluations := mockey.Mock((*adapterfunctions.FunctionContext).Evaluate).To(func(*adapterfunctions.FunctionContext, functioninterfaces.FunctionHandle, *shared.FunctionSignature, []valueinterfaces.Value) (valueinterfaces.Value, error) {
		t.Fatal("empty touchstone pipeline evaluated a function")
		return nil, merr.ErrServiceInternal
	}).Build()
	defer evaluations.UnPatch()
	for _, nullable := range []bool{false, true} {
		for _, legacy := range []string{"false", "true"} {
			t.Run(strconv.FormatBool(nullable)+"/legacy="+legacy, func(t *testing.T) {
				require.NoError(t, params.Save(legacyKey, legacy))
				helper, err := typeutil.CreateSchemaHelper(touchstone.Schema("touchstone", nullable))
				require.NoError(t, err)
				require.NoError(t, params.Save(params.ProxyCfg.EnablePredicateOptimization.Key, "false"))
				original, err := ParseExpr(helper, touchstone.Query, nil)
				require.NoError(t, err)
				require.NoError(t, params.Save(params.ProxyCfg.EnablePredicateOptimization.Key, "true"))
				imports, attempts := 0, 0
				var importGraph func(*translation.Translator, context.Context, *planpb.Expr) (*translation.ImportResult, error)
				inspect := mockey.Mock((*translation.Translator).Import).Origin(&importGraph).To(func(translator *translation.Translator, ctx context.Context, source *planpb.Expr) (*translation.ImportResult, error) {
					imports++
					result, err := importGraph(translator, ctx, source)
					require.NoError(t, err)
					counts := map[string]int{}
					var walk func(expressions.Expression)
					walk = func(node expressions.Expression) {
						switch n := node.(type) {
						case *expressions.OpaqueExpression:
							t.Fatal("touchstone operation hidden in an opaque region")
						case *expressions.CallExpression:
							runtime, err := adapterruntime.SharedRuntime()
							require.NoError(t, err)
							metadata, err := runtime.FunctionContext().Metadata(n.Handle())
							require.NoError(t, err)
							if name, exists := metadata.BuiltInName().Get(); exists {
								counts[string(name)]++
							} else {
								counts["comparison"]++
							}
						case *expressions.SpecialFormExpression:
							counts["membership"]++
						}
						for i := 0; i < node.ChildCount(); i++ {
							walk(node.ChildAt(i))
						}
					}
					walk(result.Root())
					require.Equal(t, map[string]int{string(shared.BuiltInArrayContains): 7, string(shared.BuiltInArrayContainsAny): 3, "comparison": 1, "membership": 7}, counts)
					return result, nil
				}).Build()
				defer inspect.UnPatch()
				var run func(context.Context, orchestration.AttemptInput, func(*planpb.Expr) error) (orchestration.AttemptResult, error)
				observe := mockey.Mock(orchestration.RoundTrip).Origin(&run).To(func(ctx context.Context, input orchestration.AttemptInput, validate func(*planpb.Expr) error) (orchestration.AttemptResult, error) {
					attempts++
					require.True(t, proto.Equal(original, input.Original))
					before := proto.Clone(input.Original)
					result, err := run(ctx, input, validate)
					require.NoError(t, err)
					require.Equal(t, orchestration.OutcomeTranslated, result.Outcome())
					require.True(t, proto.Equal(before, input.Original))
					return result, err
				}).Build()
				defer observe.UnPatch()
				output, err := ParseExpr(helper, touchstone.Query, nil)
				require.NoError(t, err)
				require.Equal(t, 1, attempts)
				require.Equal(t, 1, imports)
				require.True(t, proto.Equal(original, output))
			})
		}
	}
}

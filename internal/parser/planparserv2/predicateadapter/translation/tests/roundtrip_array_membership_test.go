package translation_test

import (
	"context"
	"testing"

	"github.com/lie-yan/predicate-optimization/expressions"
	"github.com/lie-yan/predicate-optimization/foundation"
	shared "github.com/lie-yan/predicate-optimization/functions"
	"github.com/lie-yan/predicate-optimization/types"
	"github.com/stretchr/testify/require"
	"google.golang.org/protobuf/proto"

	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
	adapterruntime "github.com/milvus-io/milvus/internal/parser/planparserv2/predicateadapter/runtime"
	"github.com/milvus-io/milvus/internal/parser/planparserv2/predicateadapter/schema"
	"github.com/milvus-io/milvus/internal/parser/planparserv2/predicateadapter/translation"
	"github.com/milvus-io/milvus/pkg/v3/proto/planpb"
	"github.com/milvus-io/milvus/pkg/v3/util/typeutil"
)

func arrayFixture(t *testing.T, element schemapb.DataType, nullable, elementNullable bool) (*translation.Translator, *planpb.ColumnInfo) {
	t.Helper()
	field := &schemapb.FieldSchema{FieldID: 101, Name: "items", DataType: schemapb.DataType_Array, ElementType: element, Nullable: nullable, ElementNullable: elementNullable}
	fields := []*schemapb.FieldSchema{field}
	helper, err := typeutil.CreateSchemaHelper(&schemapb.CollectionSchema{Fields: fields})
	require.NoError(t, err)
	view, err := schema.NewView(helper, fields)
	require.NoError(t, err)
	runtime, err := adapterruntime.SharedRuntime()
	require.NoError(t, err)
	translator, err := translation.New(&guardedRuntime{Runtime: runtime, provider: &noEvaluationContext{runtime.FunctionContext()}}, view, runtime.Types())
	require.NoError(t, err)
	return translator, &planpb.ColumnInfo{FieldId: 101, DataType: schemapb.DataType_Array, ElementType: element, Nullable: nullable}
}

func arraySource(column *planpb.ColumnInfo, op planpb.JSONContainsExpr_JSONOp, candidates ...*planpb.GenericValue) *planpb.Expr {
	return &planpb.Expr{Expr: &planpb.Expr_JsonContainsExpr{JsonContainsExpr: &planpb.JSONContainsExpr{ColumnInfo: column, Op: op, Elements: candidates, ElementsSameType: true}}}
}

func TestArrayMembershipMappings(t *testing.T) {
	for _, tc := range []struct {
		element   schemapb.DataType
		candidate *planpb.GenericValue
	}{
		{schemapb.DataType_Bool, &planpb.GenericValue{Val: &planpb.GenericValue_BoolVal{BoolVal: true}}},
		{schemapb.DataType_Int8, integerSource(127)},
		{schemapb.DataType_Int16, integerSource(32767)},
		{schemapb.DataType_Int32, integerSource(2147483647)},
		{schemapb.DataType_Int64, integerSource(9007199254740993)},
		{schemapb.DataType_Float, floatSource(1.5)},
		{schemapb.DataType_Double, floatSource(1.1)},
	} {
		for _, nullable := range []bool{false, true} {
			translator, column := arrayFixture(t, tc.element, nullable, false)
			for _, source := range []*planpb.Expr{
				arraySource(column, planpb.JSONContainsExpr_Contains, tc.candidate),
				arraySource(column, planpb.JSONContainsExpr_ContainsAny, tc.candidate, tc.candidate),
				arraySource(column, planpb.JSONContainsExpr_ContainsAny),
			} {
				before := proto.Clone(source)
				source.IsTemplate = true
				source.GetJsonContainsExpr().TemplateVariableName = "filled"
				imported := importSource(t, translator, source)
				call := imported.Root().(*expressions.CallExpression)
				require.Equal(t, call.ChildAt(0).Type().Nullability(), call.Type().Nullability())
				require.Equal(t, foundation.NonNull, call.ChildAt(0).Type().(*types.ArrayType).Element().Nullability())
				runtime, err := adapterruntime.SharedRuntime()
				require.NoError(t, err)
				metadata, err := runtime.FunctionContext().Metadata(call.Handle())
				require.NoError(t, err)
				require.Equal(t, shared.PropagateNull, metadata.NullPolicyAt(0))
				require.Equal(t, shared.NullPolicyUnspecified, metadata.NullPolicyAt(1))
				output, err := translator.Export(context.Background(), imported, rebuiltCall(t, call, call.ChildAt(0), call.ChildAt(1)))
				require.NoError(t, err)
				require.True(t, proto.Equal(before, output), "%s nullable=%v: %s", tc.element, nullable, output)
				output.GetJsonContainsExpr().ColumnInfo.ElementType = schemapb.DataType_None
				if len(output.GetJsonContainsExpr().Elements) > 0 {
					output.GetJsonContainsExpr().Elements[0].Val = &planpb.GenericValue_BoolVal{}
				}
				source.IsTemplate = false
				source.GetJsonContainsExpr().TemplateVariableName = ""
				require.True(t, proto.Equal(before, source), "export mutated borrowed source")
			}
		}
	}
}

func TestArrayMembershipRewrittenGraph(t *testing.T) {
	translator, column := arrayFixture(t, schemapb.DataType_Int64, true, false)
	imported := importSource(t, translator, arraySource(column, planpb.JSONContainsExpr_ContainsAny, integerSource(7), integerSource(9)))
	call := imported.Root().(*expressions.CallExpression)
	runtime, err := adapterruntime.SharedRuntime()
	require.NoError(t, err)
	integer, err := runtime.ValueFactory().NewInteger(runtime.Types().Int64NonNull, 11)
	require.NoError(t, err)
	literal, err := expressions.NewLiteralExpression(integer)
	require.NoError(t, err)
	handle, err := runtime.FunctionContext().LookupBuiltInFunction(shared.BuiltInArrayContains, []types.Type{call.ChildAt(0).Type(), literal.Type()})
	require.NoError(t, err)
	changed, err := expressions.NewCallExpression(runtime.FunctionContext(), handle, []expressions.Expression{call.ChildAt(0), literal})
	require.NoError(t, err)
	output, err := translator.Export(context.Background(), imported, logicalGraph(t, foundation.Not, changed))
	require.NoError(t, err)
	require.True(t, proto.Equal(notSource(arraySource(column, planpb.JSONContainsExpr_Contains, integerSource(11))), output))
}

func TestArrayMembershipCoverage(t *testing.T) {
	for _, tc := range []struct {
		element         schemapb.DataType
		elementNullable bool
		value           *planpb.GenericValue
	}{
		{schemapb.DataType_Int8, false, integerSource(300)},
		{schemapb.DataType_Int64, false, floatSource(1.5)},
		{schemapb.DataType_Float, false, floatSource(1.1)},
		{schemapb.DataType_VarChar, false, stringSource("x")},
		{schemapb.DataType_Int64, true, integerSource(1)},
	} {
		translator, column := arrayFixture(t, tc.element, true, tc.elementNullable)
		result, err := translator.Import(context.Background(), arraySource(column, planpb.JSONContainsExpr_Contains, tc.value))
		require.Nil(t, result)
		assertFailure(t, err, translation.UnsupportedExpression, translation.StageImport)
	}
	translator, column := arrayFixture(t, schemapb.DataType_Int64, false, false)
	for _, source := range []*planpb.Expr{
		arraySource(column, planpb.JSONContainsExpr_ContainsAll, integerSource(1)),
		arraySource(&planpb.ColumnInfo{DataType: schemapb.DataType_JSON}, planpb.JSONContainsExpr_Contains, integerSource(1)),
	} {
		_, err := translator.Import(context.Background(), source)
		assertFailure(t, err, translation.UnsupportedExpression, translation.StageImport)
	}
	_, err := translator.Import(context.Background(), arraySource(column, planpb.JSONContainsExpr_Contains))
	assertFailure(t, err, translation.UnsafeSource, translation.StageImport)
}

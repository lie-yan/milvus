package translation_test

import (
	"context"
	"math"
	"strings"
	"testing"

	"github.com/lie-yan/predicate-optimization/expressions"
	"github.com/lie-yan/predicate-optimization/foundation"
	"github.com/stretchr/testify/require"
	"google.golang.org/protobuf/proto"

	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
	"github.com/milvus-io/milvus/internal/parser/planparserv2/predicateadapter/translation"
	"github.com/milvus-io/milvus/pkg/v3/proto/planpb"
)

func stringSource(value string) *planpb.GenericValue {
	return &planpb.GenericValue{Val: &planpb.GenericValue_StringVal{StringVal: value}}
}

func termSource(field *planpb.ColumnInfo, values ...*planpb.GenericValue) *planpb.Expr {
	return &planpb.Expr{IsTemplate: true, Expr: &planpb.Expr_TermExpr{TermExpr: &planpb.TermExpr{ColumnInfo: field, Values: values, TemplateVariableName: "candidates"}}}
}

func TestTermMappings(t *testing.T) {
	for _, tc := range []struct {
		typ    schemapb.DataType
		values []*planpb.GenericValue
	}{
		{schemapb.DataType_Bool, []*planpb.GenericValue{{Val: &planpb.GenericValue_BoolVal{}}, {Val: &planpb.GenericValue_BoolVal{BoolVal: true}}}},
		{schemapb.DataType_Int8, []*planpb.GenericValue{integerSource(-128), integerSource(127)}},
		{schemapb.DataType_Int16, []*planpb.GenericValue{integerSource(-32768), integerSource(32767)}},
		{schemapb.DataType_Int32, []*planpb.GenericValue{integerSource(math.MinInt32), integerSource(math.MaxInt32)}},
		{schemapb.DataType_Int64, []*planpb.GenericValue{integerSource(math.MaxInt64), integerSource(math.MinInt64), integerSource(9007199254740993), integerSource(math.MaxInt64)}},
		{schemapb.DataType_Float, []*planpb.GenericValue{floatSource(math.Copysign(0, -1)), floatSource(1.5)}},
		{schemapb.DataType_Double, []*planpb.GenericValue{floatSource(math.Inf(1)), floatSource(math.Copysign(0, -1)), floatSource(0), floatSource(math.Inf(-1))}},
		{schemapb.DataType_VarChar, []*planpb.GenericValue{stringSource("é"), stringSource(""), stringSource("a\x00"), stringSource("z"), stringSource("é")}},
	} {
		for _, nullable := range []bool{false, true} {
			t.Run(tc.typ.String(), func(t *testing.T) {
				translator, fields := compareFixture(t, tc.typ, tc.typ, nullable)
				for _, count := range []int{1, len(tc.values)} {
					source := termSource(fields.GetCompareExpr().RightColumnInfo, tc.values[:count]...)
					for _, negated := range []bool{false, true} {
						input := source
						if negated {
							input = notSource(source)
							input.IsTemplate = true
						}
						imported := importSource(t, translator, input)
						in := imported.Root()
						if negated {
							in = in.ChildAt(0)
						}
						special := in.(*expressions.SpecialFormExpression)
						rebuilt, err := expressions.NewSpecialFormExpression(special.Form(), special.Type(), special.Children())
						require.NoError(t, err)
						var graph expressions.Expression = rebuilt
						if negated {
							graph = logicalGraph(t, foundation.Not, rebuilt)
						}
						output, err := translator.Export(context.Background(), imported, graph)
						require.NoError(t, err)
						require.True(t, proto.Equal(expectedExport(input), output))
						term := output.GetTermExpr()
						if negated {
							term = output.GetUnaryExpr().Child.GetTermExpr()
						}
						for i, value := range expectedExport(source).GetTermExpr().Values {
							if _, ok := value.Val.(*planpb.GenericValue_FloatVal); ok {
								require.Equal(t, math.Float64bits(value.GetFloatVal()), math.Float64bits(term.Values[i].GetFloatVal()))
							}
						}
					}
				}
			})
		}
	}
}

func TestTermImportCoverage(t *testing.T) {
	for _, tc := range []struct {
		name   string
		typ    schemapb.DataType
		values []*planpb.GenericValue
		kind   translation.FailureKind
	}{
		{"empty", schemapb.DataType_Int64, nil, translation.UnsupportedExpression},
		{"overflow", schemapb.DataType_Int8, []*planpb.GenericValue{integerSource(300)}, translation.UnsupportedExpression},
		{"float rounding", schemapb.DataType_Float, []*planpb.GenericValue{floatSource(0.1)}, translation.UnsupportedExpression},
		{"conversion", schemapb.DataType_Double, []*planpb.GenericValue{integerSource(1)}, translation.UnsupportedExpression},
		{"heterogeneous", schemapb.DataType_Int64, []*planpb.GenericValue{integerSource(1), stringSource("x")}, translation.UnsupportedExpression},
		{"string bound", schemapb.DataType_VarChar, []*planpb.GenericValue{stringSource(strings.Repeat("é", 51))}, translation.UnsupportedExpression},
		{"NaN", schemapb.DataType_Double, []*planpb.GenericValue{floatSource(math.NaN())}, translation.UnsupportedExpression},
		{"missing after unsupported", schemapb.DataType_Int64, []*planpb.GenericValue{stringSource("x"), nil}, translation.UnsupportedExpression},
	} {
		t.Run(tc.name, func(t *testing.T) {
			translator, fields := compareFixture(t, tc.typ, tc.typ, false)
			imported, err := translator.Import(context.Background(), termSource(fields.GetCompareExpr().LeftColumnInfo, tc.values...))
			require.Nil(t, imported)
			assertFailure(t, err, tc.kind, translation.StageImport)
		})
	}
	translator, fields := compareFixture(t, schemapb.DataType_Int64, schemapb.DataType_Int64, true)
	imported := importSource(t, translator, termSource(fields.GetCompareExpr().RightColumnInfo, integerSource(1)))
	require.Equal(t, foundation.MayNull, imported.Root().Type().Nullability())
	for _, change := range []func(*planpb.TermExpr){
		func(n *planpb.TermExpr) { n.IsInField = true },
		func(n *planpb.TermExpr) { n.ProtoReflect().SetUnknown([]byte{0xa0, 0x06, 1}) },
	} {
		source := termSource(fields.GetCompareExpr().LeftColumnInfo, integerSource(1))
		change(source.GetTermExpr())
		_, err := translator.Import(context.Background(), source)
		assertFailure(t, err, translation.UnsupportedExpression, translation.StageImport)
	}
}

func TestTermCorrespondenceAndOwnership(t *testing.T) {
	ctx := context.Background()
	translator, fields := compareFixture(t, schemapb.DataType_Int64, schemapb.DataType_Int64, false)
	source := termSource(fields.GetCompareExpr().LeftColumnInfo, integerSource(1), integerSource(2))
	imported := importSource(t, translator, source)
	graph := imported.Root().(*expressions.SpecialFormExpression)
	foreign := importSource(t, translator, source)
	for _, args := range [][]expressions.Expression{
		{foreign.Root().ChildAt(0), graph.ChildAt(1), graph.ChildAt(2)},
		{graph.ChildAt(0), graph.ChildAt(1), graph.ChildAt(0)},
	} {
		changed, err := expressions.NewSpecialFormExpression(foundation.SpecialFormIn, graph.Type(), args)
		require.NoError(t, err)
		output, err := translator.Export(ctx, imported, changed)
		require.Nil(t, output)
		if args[0] == foreign.Root().ChildAt(0) {
			assertFailure(t, err, translation.TranslationDefect, translation.StageExport)
		} else {
			assertFailure(t, err, translation.UnsupportedExpression, translation.StageExport)
		}
	}

	second := proto.Clone(source).(*planpb.Expr)
	second.IsTemplate = false
	second.GetTermExpr().TemplateVariableName = "second"
	mixed := binarySource(planpb.BinaryExpr_LogicalAnd, source, notSource(second))
	joined := importSource(t, translator, mixed)
	root := joined.Root()
	shared := logicalGraph(t, foundation.And, root.ChildAt(0), logicalGraph(t, foundation.Not, root.ChildAt(0)))
	output, err := translator.Export(ctx, joined, shared)
	require.NoError(t, err)
	require.True(t, proto.Equal(expectedExport(mixed), output))
	output.GetBinaryExpr().Left.GetTermExpr().Values[0].Val.(*planpb.GenericValue_Int64Val).Int64Val = 99
	require.Equal(t, int64(1), output.GetBinaryExpr().Right.GetUnaryExpr().Child.GetTermExpr().Values[0].GetInt64Val())
	again, err := translator.Export(ctx, joined, shared)
	require.NoError(t, err)
	require.True(t, proto.Equal(expectedExport(mixed), again))
}

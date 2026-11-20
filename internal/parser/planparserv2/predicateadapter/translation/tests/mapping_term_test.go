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
		t.Run(tc.typ.String(), func(t *testing.T) {
			translator, fields := compareFixture(t, tc.typ, tc.typ, false)
			for _, count := range []int{1, len(tc.values)} {
				source := termSource(fields.GetCompareExpr().LeftColumnInfo, tc.values[:count]...)
				for _, negated := range []bool{false, true} {
					input := source
					if negated {
						input = notSource(source)
						input.IsTemplate = true
					}
					_, imported := importSource(t, translator, input)
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
					require.NoError(t, translator.ValidateCandidate(context.Background(), imported, output))
					require.True(t, proto.Equal(input, output))
					term := output.GetTermExpr()
					if negated {
						term = output.GetUnaryExpr().Child.GetTermExpr()
					}
					for i, value := range tc.values[:count] {
						if _, ok := value.Val.(*planpb.GenericValue_FloatVal); ok {
							require.Equal(t, math.Float64bits(value.GetFloatVal()), math.Float64bits(term.Values[i].GetFloatVal()))
						}
					}
				}
			}
		})
	}
}

func TestTermAdmission(t *testing.T) {
	for _, tc := range []struct {
		name   string
		typ    schemapb.DataType
		values []*planpb.GenericValue
		kind   translation.FailureKind
	}{
		{"empty", schemapb.DataType_Int64, nil, translation.UnsupportedSource},
		{"overflow", schemapb.DataType_Int8, []*planpb.GenericValue{integerSource(300)}, translation.UnsupportedSource},
		{"float rounding", schemapb.DataType_Float, []*planpb.GenericValue{floatSource(0.1)}, translation.UnsupportedSource},
		{"conversion", schemapb.DataType_Double, []*planpb.GenericValue{integerSource(1)}, translation.UnsupportedSource},
		{"heterogeneous", schemapb.DataType_Int64, []*planpb.GenericValue{integerSource(1), stringSource("x")}, translation.UnsupportedSource},
		{"string bound", schemapb.DataType_VarChar, []*planpb.GenericValue{stringSource(strings.Repeat("é", 51))}, translation.UnsupportedSource},
		{"NaN", schemapb.DataType_Double, []*planpb.GenericValue{floatSource(math.NaN())}, translation.UnsupportedSource},
		{"missing after unsupported", schemapb.DataType_Int64, []*planpb.GenericValue{stringSource("x"), nil}, translation.UnsafeSource},
	} {
		t.Run(tc.name, func(t *testing.T) {
			translator, fields := compareFixture(t, tc.typ, tc.typ, false)
			admitted, err := translator.Admit(context.Background(), termSource(fields.GetCompareExpr().LeftColumnInfo, tc.values...))
			require.Nil(t, admitted)
			assertFailure(t, err, tc.kind, translation.StageAdmission)
		})
	}
	translator, fields := compareFixture(t, schemapb.DataType_Int64, schemapb.DataType_Int64, true)
	_, err := translator.Admit(context.Background(), termSource(fields.GetCompareExpr().RightColumnInfo, integerSource(1)))
	assertFailure(t, err, translation.UnsupportedSource, translation.StageAdmission)
	for _, change := range []func(*planpb.TermExpr){
		func(n *planpb.TermExpr) { n.IsInField = true },
		func(n *planpb.TermExpr) { n.ProtoReflect().SetUnknown([]byte{0xa0, 0x06, 1}) },
	} {
		source := termSource(fields.GetCompareExpr().LeftColumnInfo, integerSource(1))
		change(source.GetTermExpr())
		_, err := translator.Admit(context.Background(), source)
		assertFailure(t, err, translation.UnsupportedSource, translation.StageAdmission)
	}
}

func TestTermCorrespondenceAndOwnership(t *testing.T) {
	ctx := context.Background()
	translator, fields := compareFixture(t, schemapb.DataType_Int64, schemapb.DataType_Int64, false)
	source := termSource(fields.GetCompareExpr().LeftColumnInfo, integerSource(1), integerSource(2))
	_, imported := importSource(t, translator, source)
	graph := imported.Root().(*expressions.SpecialFormExpression)
	_, foreign := importSource(t, translator, source)
	for _, args := range [][]expressions.Expression{
		{graph.ChildAt(0), graph.ChildAt(2), graph.ChildAt(1)},
		{graph.ChildAt(0), graph.ChildAt(1)},
		{foreign.Root().ChildAt(0), graph.ChildAt(1), graph.ChildAt(2)},
		{graph.ChildAt(0), graph.ChildAt(1), graph.ChildAt(0)},
	} {
		changed, err := expressions.NewSpecialFormExpression(foundation.SpecialFormIn, graph.Type(), args)
		require.NoError(t, err)
		output, err := translator.Export(ctx, imported, changed)
		require.Nil(t, output)
		assertFailure(t, err, translation.TranslationDefect, translation.StageExport)
	}
	for _, change := range []func(*planpb.TermExpr){
		func(n *planpb.TermExpr) { n.Values[0], n.Values[1] = n.Values[1], n.Values[0] },
		func(n *planpb.TermExpr) { n.Values = n.Values[:1] },
		func(n *planpb.TermExpr) { n.IsInField = true },
		func(n *planpb.TermExpr) { n.TemplateVariableName = "changed" },
	} {
		candidate := proto.Clone(source).(*planpb.Expr)
		change(candidate.GetTermExpr())
		assertFailure(t, translator.ValidateCandidate(ctx, imported, candidate), translation.TranslationDefect, translation.StageCandidateValidation)
	}
	second := proto.Clone(source).(*planpb.Expr)
	second.IsTemplate = false
	second.GetTermExpr().TemplateVariableName = "second"
	mixed := binarySource(planpb.BinaryExpr_LogicalAnd, source, notSource(second))
	_, joined := importSource(t, translator, mixed)
	root := joined.Root()
	shared := logicalGraph(t, foundation.And, root.ChildAt(0), logicalGraph(t, foundation.Not, root.ChildAt(0)))
	output, err := translator.Export(ctx, joined, shared)
	require.NoError(t, err)
	require.True(t, proto.Equal(mixed, output))
	require.NoError(t, translator.ValidateCandidate(ctx, joined, output))
	output.GetBinaryExpr().Left.GetTermExpr().Values[0].Val.(*planpb.GenericValue_Int64Val).Int64Val = 99
	require.Equal(t, int64(1), output.GetBinaryExpr().Right.GetUnaryExpr().Child.GetTermExpr().Values[0].GetInt64Val())
	again, err := translator.Export(ctx, joined, shared)
	require.NoError(t, err)
	require.True(t, proto.Equal(mixed, again))
}

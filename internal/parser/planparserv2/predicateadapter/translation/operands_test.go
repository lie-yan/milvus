package translation

import (
	"context"
	"errors"
	"math"
	"testing"

	"github.com/lie-yan/predicate-optimization/expressions"
	"github.com/lie-yan/predicate-optimization/types"
	valueinterfaces "github.com/lie-yan/predicate-optimization/values/interfaces"
	"github.com/stretchr/testify/require"
	"google.golang.org/protobuf/proto"

	"github.com/milvus-io/milvus-proto/go-api/v3/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
	adapterruntime "github.com/milvus-io/milvus/internal/parser/planparserv2/predicateadapter/runtime"
	"github.com/milvus-io/milvus/internal/parser/planparserv2/predicateadapter/schema"
	"github.com/milvus-io/milvus/pkg/v3/proto/planpb"
	"github.com/milvus-io/milvus/pkg/v3/util/typeutil"
)

func operandFixture(t *testing.T) (*Translator, *schema.FieldRegistry, []*planpb.ColumnInfo) {
	t.Helper()
	fields := []*schemapb.FieldSchema{
		{FieldID: 101, Name: "i", DataType: schemapb.DataType_Int64, IsPrimaryKey: true},
		{FieldID: 102, Name: "j", DataType: schemapb.DataType_Int64},
		{FieldID: 103, Name: "b", DataType: schemapb.DataType_Bool},
		{FieldID: 104, Name: "nullable", DataType: schemapb.DataType_Int64, Nullable: true},
		{FieldID: 105, Name: "narrow", DataType: schemapb.DataType_Int8},
		{FieldID: 106, Name: "i16", DataType: schemapb.DataType_Int16},
		{FieldID: 107, Name: "i32", DataType: schemapb.DataType_Int32},
		{FieldID: 108, Name: "f32", DataType: schemapb.DataType_Float},
		{FieldID: 109, Name: "f64", DataType: schemapb.DataType_Double},
		{FieldID: 110, Name: "text", DataType: schemapb.DataType_VarChar, TypeParams: []*commonpb.KeyValuePair{{Key: "max_length", Value: "8"}}},
	}
	helper, err := typeutil.CreateSchemaHelper(&schemapb.CollectionSchema{Fields: fields})
	require.NoError(t, err)
	view, err := schema.NewView(helper, fields)
	require.NoError(t, err)
	runtime, err := adapterruntime.SharedRuntime()
	require.NoError(t, err)
	translator, err := New(runtime, view, runtime.Types())
	require.NoError(t, err)
	registry, err := schema.NewFieldRegistry(view)
	require.NoError(t, err)
	columns := make([]*planpb.ColumnInfo, len(fields))
	for i, field := range fields {
		columns[i] = &planpb.ColumnInfo{FieldId: field.FieldID, DataType: field.DataType, Nullable: field.Nullable, IsPrimaryKey: field.IsPrimaryKey}
	}
	return translator, registry, columns
}

func requireOperandFailure(t *testing.T, err error, kind FailureKind, stage Stage) {
	t.Helper()
	var failure *Failure
	require.ErrorAs(t, err, &failure)
	require.Equal(t, kind, failure.Kind())
	require.Equal(t, stage, failure.Stage())
}

func TestFieldOperandRoundTripAndRole(t *testing.T) {
	translator, registry, columns := operandFixture(t)
	var operands []fieldOperand
	var graphs []*expressions.ReferenceExpression
	for _, column := range columns[:4] {
		operand, reason, err := translator.readFieldOperand(column)
		require.NoError(t, err)
		require.Equal(t, noLimitation, reason)
		graph, err := importField(registry, operand)
		require.NoError(t, err)
		rebuilt, err := expressions.NewReferenceExpression(graph.ID(), graph.Type(), "different diagnostic name")
		require.NoError(t, err)
		output, err := exportField(registry, rebuilt)
		require.NoError(t, err)
		require.True(t, proto.Equal(column, output))
		require.NotSame(t, column, output)
		operands = append(operands, operand)
		graphs = append(graphs, graph)
	}
	// Repeated references share identity but receive independent exported metadata.
	repeated := proto.Clone(columns[0]).(*planpb.ColumnInfo)
	operand, reason, err := translator.readFieldOperand(repeated)
	require.NoError(t, err)
	require.Equal(t, noLimitation, reason)
	graph, err := importField(registry, operand)
	require.NoError(t, err)
	require.True(t, graphs[0].ID().EqualsTo(graph.ID()))
	require.Same(t, repeated, operand.source)
	first, err := exportField(registry, graphs[0])
	require.NoError(t, err)
	second, err := exportField(registry, graph)
	require.NoError(t, err)
	first.NestedPath = []string{"changed"}
	require.Empty(t, second.NestedPath)
	require.Empty(t, repeated.NestedPath)
	require.Empty(t, columns[0].NestedPath)
	// Export follows a registered graph reference rather than a source position.
	replacement, err := exportField(registry, graphs[1])
	require.NoError(t, err)
	require.True(t, proto.Equal(columns[1], replacement))
	foreign, err := schema.NewFieldRegistry(translator.view)
	require.NoError(t, err)
	foreignGraph, err := importField(foreign, operands[0])
	require.NoError(t, err)
	changedType, err := expressions.NewReferenceExpression(graph.ID(), nullableIntegerType(t), "")
	require.NoError(t, err)
	for _, changed := range []expressions.Expression{foreignGraph, changedType} {
		output, err := exportField(registry, changed)
		require.Nil(t, output)
		requireOperandFailure(t, err, TranslationDefect, StageExport)
	}
}

func TestOperandImportCoverageCoverageAndDefects(t *testing.T) {
	translator, _, columns := operandFixture(t)
	nested := proto.Clone(columns[0]).(*planpb.ColumnInfo)
	nested.NestedPath = []string{"x"}
	unknownColumn := proto.Clone(columns[0]).(*planpb.ColumnInfo)
	unknownColumn.ProtoReflect().SetUnknown([]byte{0xa0, 0x06, 1})
	for _, column := range []*planpb.ColumnInfo{nested, unknownColumn} {
		operand, reason, err := translator.readFieldOperand(column)
		require.NoError(t, err)
		require.NotEqual(t, noLimitation, reason)
		require.Nil(t, operand.source)
	}
	mismatch := proto.Clone(columns[0]).(*planpb.ColumnInfo)
	mismatch.DataType = schemapb.DataType_Bool
	for _, column := range []*planpb.ColumnInfo{nil, mismatch} {
		_, _, err := translator.readFieldOperand(column)
		requireOperandFailure(t, err, UnsafeSource, StageImport)
	}
	for _, tc := range []struct {
		value *planpb.GenericValue
		typ   types.Type
	}{
		{nil, translator.primitives.BoolNonNull},
		{&planpb.GenericValue{}, translator.primitives.Int64NonNull},
		{&planpb.GenericValue{Val: (*planpb.GenericValue_Int64Val)(nil)}, translator.primitives.Int64NonNull},
		{&planpb.GenericValue{Val: &planpb.GenericValue_BoolVal{BoolVal: false}}, translator.primitives.Int64NonNull},
		{&planpb.GenericValue{Val: &planpb.GenericValue_Int64Val{Int64Val: 0}}, translator.primitives.BoolNonNull},
	} {
		_, _, err := translator.readLiteralOperand(tc.value, tc.typ)
		requireOperandFailure(t, err, UnsafeSource, StageImport)
	}
	unknown := &planpb.GenericValue{Val: &planpb.GenericValue_Int64Val{Int64Val: 1}}
	unknown.ProtoReflect().SetUnknown([]byte{0xa0, 0x06, 1})
	_, reason, err := translator.readLiteralOperand(unknown, translator.primitives.Int64NonNull)
	require.NoError(t, err)
	require.Equal(t, excludedScopeOrVersion, reason)

}

// Public payload views deliberately hide the factory's concrete Value representation.
type boolOperandView struct {
	valueinterfaces.Value
	payload bool
}

func (v boolOperandView) Bool() bool { return v.payload }

type integerOperandView struct {
	valueinterfaces.Value
	payload int64
}

func (v integerOperandView) Integer() int64 { return v.payload }

func TestLiteralOperandExactRoundTrip(t *testing.T) {
	translator, _, _ := operandFixture(t)
	cases := []struct {
		source *planpb.GenericValue
		typ    types.Type
	}{
		{&planpb.GenericValue{Val: &planpb.GenericValue_BoolVal{BoolVal: false}}, translator.primitives.BoolNonNull},
		{&planpb.GenericValue{Val: &planpb.GenericValue_BoolVal{BoolVal: true}}, translator.primitives.BoolNonNull},
	}
	for _, n := range []int64{0, -1, math.MinInt64, math.MaxInt64, 9007199254740993} {
		cases = append(cases, struct {
			source *planpb.GenericValue
			typ    types.Type
		}{&planpb.GenericValue{Val: &planpb.GenericValue_Int64Val{Int64Val: n}}, translator.primitives.Int64NonNull})
	}
	for i, tc := range cases {
		operand, reason, err := translator.readLiteralOperand(tc.source, tc.typ)
		require.NoError(t, err)
		require.Equal(t, noLimitation, reason)
		graph, err := translator.importLiteral(operand)
		require.NoError(t, err)
		var peer valueinterfaces.Value
		switch payload := tc.source.Val.(type) {
		case *planpb.GenericValue_BoolVal:
			peer = boolOperandView{graph.Value(), payload.BoolVal}
		case *planpb.GenericValue_Int64Val:
			peer = integerOperandView{graph.Value(), payload.Int64Val}
		}
		rebuilt, err := expressions.NewLiteralExpression(peer)
		require.NoError(t, err)
		output, err := exportScalar(rebuilt)
		require.NoError(t, err)
		require.True(t, proto.Equal(tc.source, output))
		second, err := exportScalar(graph)
		require.NoError(t, err)
		if i == 0 || i == 2 { // One ownership check per Bool/Int64 arm.
			require.NotSame(t, output, second)
			require.NotSame(t, output.Val, second.Val)
		}
		switch payload := output.Val.(type) {
		case *planpb.GenericValue_BoolVal:
			payload.BoolVal = !payload.BoolVal
		case *planpb.GenericValue_Int64Val:
			payload.Int64Val ^= 1
		}
		require.True(t, proto.Equal(tc.source, second))
	}
}

func TestScalarExportRejectsNullableValues(t *testing.T) {
	translator, _, _ := operandFixture(t)
	nullable, err := translator.factory.NewInteger(nullableIntegerType(t), 7)
	require.NoError(t, err)
	null, err := translator.factory.NewNull(nullableIntegerType(t))
	require.NoError(t, err)
	for _, value := range []valueinterfaces.Value{nullable, null} {
		graph, err := expressions.NewLiteralExpression(value)
		require.NoError(t, err)
		output, err := exportScalar(graph)
		require.Nil(t, output)
		requireOperandFailure(t, err, UnsupportedExpression, StageExport)
	}
}

// Factory failure after admission must retain its cause instead of becoming coverage.
type failingOperandFactory struct {
	valueinterfaces.ValueFactory
	cause error
}

func (f failingOperandFactory) NewInteger(*types.IntType, int64) (valueinterfaces.Value, error) {
	return nil, f.cause
}

func TestOperandConstructionFailure(t *testing.T) {
	translator, _, columns := operandFixture(t)
	source := &planpb.Expr{Expr: &planpb.Expr_BinaryExpr{BinaryExpr: &planpb.BinaryExpr{
		Op:   planpb.BinaryExpr_LogicalAnd,
		Left: &planpb.Expr{Expr: &planpb.Expr_AlwaysTrueExpr{AlwaysTrueExpr: &planpb.AlwaysTrueExpr{}}},
		Right: &planpb.Expr{Expr: &planpb.Expr_UnaryRangeExpr{UnaryRangeExpr: &planpb.UnaryRangeExpr{
			ColumnInfo: columns[0], Op: planpb.OpType_Equal,
			Value: &planpb.GenericValue{Val: &planpb.GenericValue_Int64Val{Int64Val: 1}},
		}}},
	}}}
	before := proto.Clone(source)
	factory := translator.factory
	cause := errors.New("injected factory failure")
	translator.factory = failingOperandFactory{ValueFactory: factory, cause: cause}
	imported, err := translator.Import(context.Background(), source)
	require.Nil(t, imported)
	require.ErrorIs(t, err, cause)
	requireOperandFailure(t, err, TranslationDefect, StageImport)
	require.True(t, proto.Equal(before, source))
	translator.factory = factory
	imported, err = translator.Import(context.Background(), source)
	require.NoError(t, err)
	output, err := translator.Export(context.Background(), imported, imported.Root())
	require.NoError(t, err)
	require.True(t, proto.Equal(before, output))
}

func TestExtendedScalarOperandCodecs(t *testing.T) {
	translator, registry, columns := operandFixture(t)
	expected := []types.Type{translator.primitives.Int8NonNull, translator.primitives.Int16NonNull, translator.primitives.Int32NonNull, translator.primitives.Float32NonNull, translator.primitives.Float64NonNull}
	bound := uint64(8)
	varchar, err := types.NewVarCharType(&bound, translator.primitives.BoolNonNull.Nullability())
	require.NoError(t, err)
	expected = append(expected, varchar)
	for i, column := range columns[4:] {
		operand, reason, err := translator.readFieldOperand(column)
		require.NoError(t, err)
		require.Equal(t, noLimitation, reason)
		require.True(t, expected[i].EqualsTo(operand.typ))
		graph, err := importField(registry, operand)
		require.NoError(t, err)
		output, err := exportField(registry, graph)
		require.NoError(t, err)
		require.True(t, proto.Equal(column, output))
	}
	roundTrip := func(source *planpb.GenericValue, typ types.Type) {
		t.Helper()
		operand, reason, err := translator.readLiteralOperand(source, typ)
		require.NoError(t, err)
		require.Equal(t, noLimitation, reason)
		graph, err := translator.importLiteral(operand)
		require.NoError(t, err)
		require.True(t, typ.EqualsTo(graph.Type()))
		output, err := exportScalar(graph)
		require.NoError(t, err)
		require.True(t, proto.Equal(source, output))
		if v, ok := source.Val.(*planpb.GenericValue_FloatVal); ok {
			require.Equal(t, math.Float64bits(v.FloatVal), math.Float64bits(output.GetFloatVal()))
		}
	}
	for _, typ := range []*types.IntType{translator.primitives.Int8NonNull, translator.primitives.Int16NonNull, translator.primitives.Int32NonNull} {
		limit := int64(1) << (typ.IntegerWidth() - 1)
		for _, value := range []int64{-limit, 0, limit - 1} {
			roundTrip(&planpb.GenericValue{Val: &planpb.GenericValue_Int64Val{Int64Val: value}}, typ)
		}
		for _, value := range []int64{-limit - 1, limit} {
			_, reason, err := translator.readLiteralOperand(&planpb.GenericValue{Val: &planpb.GenericValue_Int64Val{Int64Val: value}}, typ)
			require.NoError(t, err)
			require.Equal(t, excludedDomain, reason)
		}
	}
	for _, typ := range []*types.FloatType{translator.primitives.Float32NonNull, translator.primitives.Float64NonNull} {
		for _, value := range []float64{0, math.Copysign(0, -1), 1.5, math.Inf(1), math.Inf(-1), float64(math.SmallestNonzeroFloat32), float64(math.MaxFloat32)} {
			roundTrip(&planpb.GenericValue{Val: &planpb.GenericValue_FloatVal{FloatVal: value}}, typ)
		}
	}
	roundTrip(&planpb.GenericValue{Val: &planpb.GenericValue_FloatVal{FloatVal: 0.1}}, translator.primitives.Float64NonNull)
	for _, value := range []string{"", "hello", "éééé", "a\x00b", "\xff"} {
		roundTrip(&planpb.GenericValue{Val: &planpb.GenericValue_StringVal{StringVal: value}}, varchar)
	}
	unbounded, err := types.NewVarCharType(nil, varchar.Nullability())
	require.NoError(t, err)
	roundTrip(&planpb.GenericValue{Val: &planpb.GenericValue_StringVal{StringVal: "longer than the field bound"}}, unbounded)
	for _, tc := range []struct {
		value *planpb.GenericValue
		typ   types.Type
	}{
		{&planpb.GenericValue{Val: &planpb.GenericValue_Int64Val{Int64Val: 300}}, translator.primitives.Int8NonNull},
		{&planpb.GenericValue{Val: &planpb.GenericValue_FloatVal{FloatVal: 0.1}}, translator.primitives.Float32NonNull},
		{&planpb.GenericValue{Val: &planpb.GenericValue_FloatVal{FloatVal: math.MaxFloat64}}, translator.primitives.Float32NonNull},
		{&planpb.GenericValue{Val: &planpb.GenericValue_FloatVal{FloatVal: math.NaN()}}, translator.primitives.Float64NonNull},
		{&planpb.GenericValue{Val: &planpb.GenericValue_StringVal{StringVal: "ééééé"}}, varchar},
	} {
		_, reason, err := translator.readLiteralOperand(tc.value, tc.typ)
		require.NoError(t, err)
		require.Equal(t, excludedDomain, reason)
	}
	// Malformed published VARCHAR bounds are source defects, not unsupported query literals.
	for _, bound := range []string{"", "bad", "0"} {
		fresh, _, cols := operandFixture(t)
		field, supported, err := fresh.view.CheckColumn(cols[9])
		require.NoError(t, err)
		require.True(t, supported)
		if bound == "" {
			field.TypeParams = nil
		} else {
			field.TypeParams = []*commonpb.KeyValuePair{{Key: "max_length", Value: bound}}
		}
		_, _, err = fresh.readFieldOperand(cols[9])
		requireOperandFailure(t, err, UnsafeSource, StageImport)
	}
}

func TestExtendedLiteralCorrespondence(t *testing.T) {
	translator, _, _ := operandFixture(t)
	negativeZero := &planpb.GenericValue{Val: &planpb.GenericValue_FloatVal{FloatVal: math.Copysign(0, -1)}}
	operand, _, err := translator.readLiteralOperand(negativeZero, translator.primitives.Float64NonNull)
	require.NoError(t, err)
	graph, err := translator.importLiteral(operand)
	require.NoError(t, err)
	originalOutput, err := exportScalar(graph)
	require.NoError(t, err)
	require.NotSame(t, negativeZero.Val, originalOutput.Val)
	require.Equal(t, math.Float64bits(negativeZero.GetFloatVal()), math.Float64bits(originalOutput.GetFloatVal()))
	positiveZero, err := translator.factory.NewFloat(translator.primitives.Float64NonNull, 0)
	require.NoError(t, err)
	changed, err := expressions.NewLiteralExpression(positiveZero)
	require.NoError(t, err)
	output, err := exportScalar(changed)
	require.NoError(t, err)
	require.False(t, math.Signbit(output.GetFloatVal()))
	varchar, err := types.NewVarCharType(nil, translator.primitives.BoolNonNull.Nullability())
	require.NoError(t, err)
	text := &planpb.GenericValue{Val: &planpb.GenericValue_StringVal{StringVal: "text"}}
	stringOperand, _, err := translator.readLiteralOperand(text, varchar)
	require.NoError(t, err)
	graph, err = translator.importLiteral(stringOperand)
	require.NoError(t, err)
	originalOutput, err = exportScalar(graph)
	require.NoError(t, err)
	require.NotSame(t, text.Val, originalOutput.Val)
	wrong, err := translator.factory.NewVarChar(varchar, "other")
	require.NoError(t, err)
	changed, err = expressions.NewLiteralExpression(wrong)
	require.NoError(t, err)
	output, err = exportScalar(changed)
	require.NoError(t, err)
	require.Equal(t, "other", output.GetStringVal())
	for _, tc := range []struct {
		value *planpb.GenericValue
		typ   types.Type
	}{
		{text, translator.primitives.Float64NonNull}, {negativeZero, varchar},
	} {
		_, _, err := translator.readLiteralOperand(tc.value, tc.typ)
		requireOperandFailure(t, err, UnsafeSource, StageImport)
	}
}

func nullableIntegerType(t *testing.T) *types.IntType {
	t.Helper()
	rt, err := adapterruntime.SharedRuntime()
	require.NoError(t, err)
	return rt.Types().Int64MayNull
}

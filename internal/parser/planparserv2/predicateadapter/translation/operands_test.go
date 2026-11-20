package translation

import (
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
	for _, column := range columns[:3] {
		operand, reason, err := translator.admitField(column)
		require.NoError(t, err)
		require.Equal(t, noLimitation, reason)
		graph, err := importField(registry, operand)
		require.NoError(t, err)
		rebuilt, err := expressions.NewReferenceExpression(graph.ID(), graph.Type(), "different diagnostic name")
		require.NoError(t, err)
		require.NoError(t, checkField(registry, operand, rebuilt, StageExport))
		output, err := exportField(registry, operand, rebuilt)
		require.NoError(t, err)
		require.NoError(t, checkFieldCandidate(operand, output))
		require.True(t, proto.Equal(column, output))
		require.NotSame(t, column, output)
		output.IsPrimaryKey = !output.IsPrimaryKey
		requireOperandFailure(t, checkFieldCandidate(operand, output), TranslationDefect, StageCandidateValidation)
		operands = append(operands, operand)
		graphs = append(graphs, graph)
	}
	// A second source position reuses field identity but owns its reconstruction association.
	repeated := proto.Clone(columns[0]).(*planpb.ColumnInfo)
	operand, reason, err := translator.admitField(repeated)
	require.NoError(t, err)
	require.Equal(t, noLimitation, reason)
	graph, err := importField(registry, operand)
	require.NoError(t, err)
	require.True(t, graphs[0].ID().EqualsTo(graph.ID()))
	require.Same(t, repeated, operand.source)
	first, err := exportField(registry, operands[0], graphs[0])
	require.NoError(t, err)
	second, err := exportField(registry, operand, graph)
	require.NoError(t, err)
	first.NestedPath = []string{"changed"}
	require.Empty(t, second.NestedPath)
	require.Empty(t, repeated.NestedPath)
	require.Empty(t, columns[0].NestedPath)
	foreign, err := schema.NewFieldRegistry(translator.view)
	require.NoError(t, err)
	foreignGraph, err := importField(foreign, operands[0])
	require.NoError(t, err)
	changedType, err := expressions.NewReferenceExpression(graph.ID(), translator.primitives.Int64MayNull, "")
	require.NoError(t, err)
	requireOperandFailure(t, checkField(registry, operands[0], foreignGraph, StageExport), TranslationDefect, StageExport)
	for _, changed := range []expressions.Expression{graphs[1], graphs[2], foreignGraph, changedType} {
		output, err := exportField(registry, operands[0], changed)
		require.Nil(t, output)
		requireOperandFailure(t, err, TranslationDefect, StageExport)
	}
}

func TestOperandAdmissionCoverageAndDefects(t *testing.T) {
	translator, _, columns := operandFixture(t)
	nested := proto.Clone(columns[0]).(*planpb.ColumnInfo)
	nested.NestedPath = []string{"x"}
	unknownColumn := proto.Clone(columns[0]).(*planpb.ColumnInfo)
	unknownColumn.ProtoReflect().SetUnknown([]byte{0xa0, 0x06, 1})
	for _, column := range []*planpb.ColumnInfo{columns[3], nested, unknownColumn} {
		operand, reason, err := translator.admitField(column)
		require.NoError(t, err)
		require.NotEqual(t, noLimitation, reason)
		require.Nil(t, operand.source)
	}
	mismatch := proto.Clone(columns[0]).(*planpb.ColumnInfo)
	mismatch.DataType = schemapb.DataType_Bool
	for _, column := range []*planpb.ColumnInfo{nil, mismatch} {
		_, _, err := translator.admitField(column)
		requireOperandFailure(t, err, UnsafeSource, StageAdmission)
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
		_, _, err := translator.admitLiteral(tc.value, tc.typ)
		requireOperandFailure(t, err, UnsafeSource, StageAdmission)
	}
	unknown := &planpb.GenericValue{Val: &planpb.GenericValue_Int64Val{Int64Val: 1}}
	unknown.ProtoReflect().SetUnknown([]byte{0xa0, 0x06, 1})
	_, reason, err := translator.admitLiteral(unknown, translator.primitives.Int64NonNull)
	require.NoError(t, err)
	require.Equal(t, excludedScopeOrVersion, reason)
	plain := &planpb.GenericValue{Val: &planpb.GenericValue_Int64Val{Int64Val: 1}}
	for _, typ := range []types.Type{translator.primitives.Int64MayNull} {
		_, reason, err := translator.admitLiteral(plain, typ)
		require.NoError(t, err)
		require.Equal(t, excludedDomain, reason)
	}
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
		operand, reason, err := translator.admitLiteral(tc.source, tc.typ)
		require.NoError(t, err)
		require.Equal(t, noLimitation, reason)
		graph, err := translator.importLiteral(operand)
		require.NoError(t, err)
		require.NoError(t, checkLiteral(operand, graph, StageExport))
		var peer valueinterfaces.Value
		switch payload := tc.source.Val.(type) {
		case *planpb.GenericValue_BoolVal:
			peer = boolOperandView{graph.Value(), payload.BoolVal}
		case *planpb.GenericValue_Int64Val:
			peer = integerOperandView{graph.Value(), payload.Int64Val}
		}
		rebuilt, err := expressions.NewLiteralExpression(peer)
		require.NoError(t, err)
		output, err := exportLiteral(operand, rebuilt)
		require.NoError(t, err)
		require.NoError(t, checkLiteralCandidate(operand, output))
		require.True(t, proto.Equal(tc.source, output))
		second, err := exportLiteral(operand, graph)
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
		requireOperandFailure(t, checkLiteralCandidate(operand, output), TranslationDefect, StageCandidateValidation)
	}
}

func TestLiteralOperandRejectsChangedGraphAndCandidate(t *testing.T) {
	translator, _, _ := operandFixture(t)
	source := &planpb.GenericValue{Val: &planpb.GenericValue_Int64Val{Int64Val: 9007199254740993}}
	operand, _, err := translator.admitLiteral(source, translator.primitives.Int64NonNull)
	require.NoError(t, err)
	changed, err := translator.factory.NewInteger(translator.primitives.Int64NonNull, 9007199254740992)
	require.NoError(t, err)
	narrow, err := translator.factory.NewInteger(translator.primitives.Int32NonNull, 1)
	require.NoError(t, err)
	nullable, err := translator.factory.NewInteger(translator.primitives.Int64MayNull, 9007199254740993)
	require.NoError(t, err)
	null, err := translator.factory.NewNull(translator.primitives.Int64MayNull)
	require.NoError(t, err)
	for _, value := range []valueinterfaces.Value{changed, narrow, nullable, null} {
		changedGraph, err := expressions.NewLiteralExpression(value)
		require.NoError(t, err)
		requireOperandFailure(t, checkLiteral(operand, changedGraph, StageExport), TranslationDefect, StageExport)
		output, err := exportLiteral(operand, changedGraph)
		require.Nil(t, output)
		requireOperandFailure(t, err, TranslationDefect, StageExport)
	}
	unknown := proto.Clone(source).(*planpb.GenericValue)
	unknown.ProtoReflect().SetUnknown([]byte{0xa0, 0x06, 1})
	for _, candidate := range []*planpb.GenericValue{nil, {}, {Val: &planpb.GenericValue_BoolVal{}}, unknown} {
		requireOperandFailure(t, checkLiteralCandidate(operand, candidate), TranslationDefect, StageCandidateValidation)
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
	translator, _, _ := operandFixture(t)
	source := &planpb.GenericValue{Val: &planpb.GenericValue_Int64Val{Int64Val: 1}}
	operand, _, err := translator.admitLiteral(source, translator.primitives.Int64NonNull)
	require.NoError(t, err)
	cause := errors.New("injected factory failure")
	translator.factory = failingOperandFactory{ValueFactory: translator.factory, cause: cause}
	graph, err := translator.importLiteral(operand)
	require.Nil(t, graph)
	require.ErrorIs(t, err, cause)
	requireOperandFailure(t, err, TranslationDefect, StageImport)
}

func TestExtendedScalarOperandCodecs(t *testing.T) {
	translator, registry, columns := operandFixture(t)
	expected := []types.Type{translator.primitives.Int8NonNull, translator.primitives.Int16NonNull, translator.primitives.Int32NonNull, translator.primitives.Float32NonNull, translator.primitives.Float64NonNull}
	bound := uint64(8)
	varchar, err := types.NewVarCharType(&bound, translator.primitives.BoolNonNull.Nullability())
	require.NoError(t, err)
	expected = append(expected, varchar)
	for i, column := range columns[4:] {
		operand, reason, err := translator.admitField(column)
		require.NoError(t, err)
		require.Equal(t, noLimitation, reason)
		require.True(t, expected[i].EqualsTo(operand.typ))
		graph, err := importField(registry, operand)
		require.NoError(t, err)
		output, err := exportField(registry, operand, graph)
		require.NoError(t, err)
		require.True(t, proto.Equal(column, output))
		require.NoError(t, checkFieldCandidate(operand, output))
	}
	roundTrip := func(source *planpb.GenericValue, typ types.Type) {
		t.Helper()
		operand, reason, err := translator.admitLiteral(source, typ)
		require.NoError(t, err)
		require.Equal(t, noLimitation, reason)
		graph, err := translator.importLiteral(operand)
		require.NoError(t, err)
		require.True(t, typ.EqualsTo(graph.Type()))
		output, err := exportLiteral(operand, graph)
		require.NoError(t, err)
		require.NoError(t, checkLiteralCandidate(operand, output))
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
			_, reason, err := translator.admitLiteral(&planpb.GenericValue{Val: &planpb.GenericValue_Int64Val{Int64Val: value}}, typ)
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
		_, reason, err := translator.admitLiteral(tc.value, tc.typ)
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
		_, _, err = fresh.admitField(cols[9])
		requireOperandFailure(t, err, UnsafeSource, StageAdmission)
	}

}

func TestExtendedLiteralCorrespondence(t *testing.T) {
	translator, _, _ := operandFixture(t)
	negativeZero := &planpb.GenericValue{Val: &planpb.GenericValue_FloatVal{FloatVal: math.Copysign(0, -1)}}
	operand, _, err := translator.admitLiteral(negativeZero, translator.primitives.Float64NonNull)
	require.NoError(t, err)
	graph, err := translator.importLiteral(operand)
	require.NoError(t, err)
	originalOutput, err := exportLiteral(operand, graph)
	require.NoError(t, err)
	require.NotSame(t, negativeZero.Val, originalOutput.Val)
	positiveZero, err := translator.factory.NewFloat(translator.primitives.Float64NonNull, 0)
	require.NoError(t, err)
	changed, err := expressions.NewLiteralExpression(positiveZero)
	require.NoError(t, err)
	output, err := exportLiteral(operand, changed)
	require.Nil(t, output)
	requireOperandFailure(t, err, TranslationDefect, StageExport)
	requireOperandFailure(t, checkLiteralCandidate(operand, &planpb.GenericValue{Val: &planpb.GenericValue_FloatVal{FloatVal: 0}}), TranslationDefect, StageCandidateValidation)
	varchar, err := types.NewVarCharType(nil, translator.primitives.BoolNonNull.Nullability())
	require.NoError(t, err)
	text := &planpb.GenericValue{Val: &planpb.GenericValue_StringVal{StringVal: "text"}}
	stringOperand, _, err := translator.admitLiteral(text, varchar)
	require.NoError(t, err)
	graph, err = translator.importLiteral(stringOperand)
	require.NoError(t, err)
	originalOutput, err = exportLiteral(stringOperand, graph)
	require.NoError(t, err)
	require.NotSame(t, text.Val, originalOutput.Val)
	wrong, err := translator.factory.NewVarChar(varchar, "other")
	require.NoError(t, err)
	changed, err = expressions.NewLiteralExpression(wrong)
	require.NoError(t, err)
	output, err = exportLiteral(stringOperand, changed)
	require.Nil(t, output)
	requireOperandFailure(t, err, TranslationDefect, StageExport)
	for _, tc := range []struct {
		value *planpb.GenericValue
		typ   types.Type
	}{
		{text, translator.primitives.Float64NonNull}, {negativeZero, varchar},
	} {
		_, _, err := translator.admitLiteral(tc.value, tc.typ)
		requireOperandFailure(t, err, UnsafeSource, StageAdmission)
	}
}

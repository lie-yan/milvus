package translation

import (
	"math"
	"slices"
	"strconv"

	"github.com/lie-yan/predicate-optimization/expressions"
	"github.com/lie-yan/predicate-optimization/foundation"
	"github.com/lie-yan/predicate-optimization/types"
	valueinterfaces "github.com/lie-yan/predicate-optimization/values/interfaces"
	"google.golang.org/protobuf/proto"

	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
	"github.com/milvus-io/milvus/internal/parser/planparserv2/predicateadapter/schema"
	"github.com/milvus-io/milvus/pkg/v3/proto/planpb"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
	"github.com/milvus-io/milvus/pkg/v3/util/typeutil"
)

// Admitted embedded field operand, independent of whole-Expr occurrence records.
type fieldOperand struct {
	// Borrowed metadata for this source position, immutable throughout the attempt.
	source *planpb.ColumnInfo
	// Shared full Type established from the pinned schema.
	typ types.Type
}

// Admitted embedded scalar operand; no Value or graph allocation during admission.
type literalOperand struct {
	// Borrowed exact oneof payload, immutable throughout the attempt.
	source *planpb.GenericValue
	// Expected full Type supplied by the enclosing mapping.
	typ types.Type
}

// Separates native field coverage from the schema view's broader validation support.
func (t *Translator) admitField(source *planpb.ColumnInfo) (fieldOperand, limitation, error) {
	field, supported, err := t.view.CheckColumn(source)
	if err != nil {
		return fieldOperand{}, noLimitation, newFailure(UnsafeSource, StageAdmission, err)
	}
	if !supported {
		return fieldOperand{}, excludedScopeOrVersion, nil
	}
	if field.GetNullable() {
		return fieldOperand{}, excludedDomain, nil
	}
	var typ types.Type
	switch field.GetDataType() {
	case schemapb.DataType_Bool:
		typ = t.primitives.BoolNonNull
	case schemapb.DataType_Int8:
		typ = t.primitives.Int8NonNull
	case schemapb.DataType_Int16:
		typ = t.primitives.Int16NonNull
	case schemapb.DataType_Int32:
		typ = t.primitives.Int32NonNull
	case schemapb.DataType_Int64:
		typ = t.primitives.Int64NonNull
	case schemapb.DataType_Float:
		typ = t.primitives.Float32NonNull
	case schemapb.DataType_Double:
		typ = t.primitives.Float64NonNull
	case schemapb.DataType_VarChar:
		// Match schema parameter lookup without appending into borrowed slice storage.
		params := typeutil.NewKvPairs(slices.Concat(field.GetIndexParams(), field.GetTypeParams()))
		bound, err := params.Get("max_length")
		if err != nil {
			return fieldOperand{}, noLimitation, newFailure(UnsafeSource, StageAdmission, merr.WrapErrServiceInternalErr(err, "predicate translation: missing VARCHAR bound"))
		}
		maxBytes, err := strconv.ParseUint(bound, 10, 64)
		if err != nil {
			return fieldOperand{}, noLimitation, newFailure(UnsafeSource, StageAdmission, merr.WrapErrServiceInternalErr(err, "predicate translation: invalid VARCHAR bound"))
		}
		if maxBytes == 0 {
			return fieldOperand{}, noLimitation, sourceDefect("invalid zero VARCHAR schema bound")
		}
		typ, err = types.NewVarCharType(&maxBytes, foundation.NonNull)
		if err != nil {
			return fieldOperand{}, noLimitation, newFailure(UnsafeSource, StageAdmission, err)
		}
	default:
		return fieldOperand{}, excludedDomain, nil
	}
	return fieldOperand{source: source, typ: typ}, noLimitation, nil
}

// Checks actual oneof arms before any default getter or Value construction.
func (t *Translator) admitLiteral(source *planpb.GenericValue, typ types.Type) (literalOperand, limitation, error) {
	if source == nil || source.Val == nil {
		return literalOperand{}, noLimitation, sourceDefect("missing literal payload")
	}
	if len(source.ProtoReflect().GetUnknown()) != 0 {
		return literalOperand{}, excludedScopeOrVersion, nil
	}
	if typ == nil {
		return literalOperand{}, noLimitation, sourceDefect("missing literal Type")
	}
	if typ.Nullability() != foundation.NonNull {
		return literalOperand{}, excludedDomain, nil
	}
	switch scalar := typ.(type) {
	case *types.BoolType:
		value, ok := source.Val.(*planpb.GenericValue_BoolVal)
		if !ok || value == nil {
			return literalOperand{}, noLimitation, sourceDefect("expected Bool literal arm")
		}
	case *types.IntType:
		value, ok := source.Val.(*planpb.GenericValue_Int64Val)
		if !ok || value == nil {
			return literalOperand{}, noLimitation, sourceDefect("expected integer literal arm")
		}
		if scalar.IntegerWidth() != types.Int64 {
			limit := int64(1) << (scalar.IntegerWidth() - 1)
			if value.Int64Val < -limit || value.Int64Val >= limit {
				return literalOperand{}, excludedDomain, nil
			}
		}
	case *types.FloatType:
		value, ok := source.Val.(*planpb.GenericValue_FloatVal)
		if !ok || value == nil {
			return literalOperand{}, noLimitation, sourceDefect("expected floating literal arm")
		}
		// NaN canonicalization and Float32 rounding would lose source payload information.
		if math.IsNaN(value.FloatVal) || (scalar.FloatWidth() == types.Float32 && float64(float32(value.FloatVal)) != value.FloatVal) {
			return literalOperand{}, excludedDomain, nil
		}
	case *types.VarCharType:
		value, ok := source.Val.(*planpb.GenericValue_StringVal)
		if !ok || value == nil {
			return literalOperand{}, noLimitation, sourceDefect("expected string literal arm")
		}
		if bound, bounded := scalar.MaxBytes(); bounded && uint64(len(value.StringVal)) > bound {
			return literalOperand{}, excludedDomain, nil
		}
	default:
		return literalOperand{}, excludedDomain, nil
	}

	return literalOperand{source: source, typ: typ}, noLimitation, nil
}

// Binds only an admitted operand; a later coverage miss is an import invariant failure.
func importField(registry *schema.FieldRegistry, operand fieldOperand) (*expressions.ReferenceExpression, error) {
	binding, supported, err := registry.BindColumn(operand.source, operand.typ)
	if err != nil {
		return nil, newFailure(TranslationDefect, StageImport, err)
	}
	if !supported {
		return nil, translationDefect(StageImport, "admitted field no longer bindable")
	}
	graph, err := expressions.NewReferenceExpression(binding.ColumnID(), binding.Type(), binding.Field().GetName())
	if err != nil {
		return nil, newFailure(TranslationDefect, StageImport, err)
	}
	return graph, nil
}

// Copies exact scalar payloads through the selected factory without coercion or evaluation.
func (t *Translator) importLiteral(operand literalOperand) (*expressions.LiteralExpression, error) {
	var value valueinterfaces.Value
	var err error
	switch payload := operand.source.Val.(type) {
	case *planpb.GenericValue_BoolVal:
		value, err = t.factory.NewBool(operand.typ.(*types.BoolType), payload.BoolVal)
	case *planpb.GenericValue_Int64Val:
		value, err = t.factory.NewInteger(operand.typ.(*types.IntType), payload.Int64Val)
	case *planpb.GenericValue_FloatVal:
		value, err = t.factory.NewFloat(operand.typ.(*types.FloatType), payload.FloatVal)
	case *planpb.GenericValue_StringVal:
		value, err = t.factory.NewVarChar(operand.typ.(*types.VarCharType), payload.StringVal)
	default:
		return nil, translationDefect(StageImport, "admitted literal changed arm")
	}
	if err != nil {
		return nil, newFailure(TranslationDefect, StageImport, err)
	}
	graph, err := expressions.NewLiteralExpression(value)
	if err != nil {
		return nil, newFailure(TranslationDefect, StageImport, err)
	}
	return graph, nil
}

// Checks registry ownership and the expected field role, not just membership in the registry.
func checkField(registry *schema.FieldRegistry, operand fieldOperand, graph expressions.Expression, stage Stage) error {
	reference, ok := graph.(*expressions.ReferenceExpression)
	if !ok || reference == nil {
		return translationDefect(stage, "expected field reference")
	}
	binding, err := registry.ResolveColumn(reference.ID(), reference.Type())
	if err != nil {
		return newFailure(TranslationDefect, stage, err)
	}
	if !operand.typ.EqualsTo(reference.Type()) || binding.Field().GetFieldID() != operand.source.FieldId {
		return translationDefect(stage, "reference changed field role or Type")
	}
	return nil
}

// Requires exact Type and payload correspondence through public scalar interfaces.
func checkLiteral(operand literalOperand, graph expressions.Expression, stage Stage) error {
	literal, ok := graph.(*expressions.LiteralExpression)
	if !ok || literal == nil || !operand.typ.EqualsTo(literal.Type()) || literal.Value().IsNull() {
		return translationDefect(stage, "literal changed shape, Type or null state")
	}
	switch source := operand.source.Val.(type) {
	case *planpb.GenericValue_BoolVal:
		value, ok := literal.Value().(valueinterfaces.BoolValue)
		if !ok || value.Bool() != source.BoolVal {
			return translationDefect(stage, "Bool literal changed payload")
		}
	case *planpb.GenericValue_Int64Val:
		value, ok := literal.Value().(valueinterfaces.IntegerValue)
		if !ok || value.Integer() != source.Int64Val {
			return translationDefect(stage, "Int64 literal changed payload")
		}
	case *planpb.GenericValue_FloatVal:
		value, ok := literal.Value().(valueinterfaces.FloatValue)
		if !ok || math.Float64bits(value.Float()) != math.Float64bits(source.FloatVal) {
			return translationDefect(stage, "floating literal changed payload")
		}
	case *planpb.GenericValue_StringVal:
		value, ok := literal.Value().(valueinterfaces.VarCharValue)
		if !ok || value.VarChar() != source.StringVal {
			return translationDefect(stage, "string literal changed payload")
		}
	default:
		return translationDefect(stage, "missing admitted scalar arm")
	}
	return nil
}

// Resolves the actual graph before copying this occurrence's metadata, never the first binding's metadata.
func exportField(registry *schema.FieldRegistry, operand fieldOperand, graph expressions.Expression) (*planpb.ColumnInfo, error) {
	if err := checkField(registry, operand, graph, StageExport); err != nil {
		return nil, err
	}
	return proto.Clone(operand.source).(*planpb.ColumnInfo), nil
}

// Allocates a fresh exact protobuf arm from the checked actual graph payload.
func exportLiteral(operand literalOperand, graph expressions.Expression) (*planpb.GenericValue, error) {
	if err := checkLiteral(operand, graph, StageExport); err != nil {
		return nil, err
	}
	value := graph.(*expressions.LiteralExpression).Value()
	result := &planpb.GenericValue{}
	switch operand.source.Val.(type) {
	case *planpb.GenericValue_BoolVal:
		result.Val = &planpb.GenericValue_BoolVal{BoolVal: value.(valueinterfaces.BoolValue).Bool()}
	case *planpb.GenericValue_Int64Val:
		result.Val = &planpb.GenericValue_Int64Val{Int64Val: value.(valueinterfaces.IntegerValue).Integer()}
	case *planpb.GenericValue_FloatVal:
		result.Val = &planpb.GenericValue_FloatVal{FloatVal: value.(valueinterfaces.FloatValue).Float()}
	case *planpb.GenericValue_StringVal:
		result.Val = &planpb.GenericValue_StringVal{StringVal: value.(valueinterfaces.VarCharValue).VarChar()}
	}
	return result, nil
}

// Independently checks emitted field metadata against its admitted source position.
func checkFieldCandidate(operand fieldOperand, candidate *planpb.ColumnInfo) error {
	if candidate == nil || len(candidate.ProtoReflect().GetUnknown()) != 0 || !proto.Equal(operand.source, candidate) {
		return translationDefect(StageCandidateValidation, "field candidate changed metadata")
	}
	return nil
}

// Independently checks emitted scalar arms and payloads without invoking export or the factory.
func checkLiteralCandidate(operand literalOperand, candidate *planpb.GenericValue) error {
	if candidate == nil || len(candidate.ProtoReflect().GetUnknown()) != 0 || !proto.Equal(operand.source, candidate) {
		return translationDefect(StageCandidateValidation, "literal candidate changed content")
	}
	if source, ok := operand.source.Val.(*planpb.GenericValue_FloatVal); ok {
		value, ok := candidate.Val.(*planpb.GenericValue_FloatVal)
		if !ok || value == nil || math.Float64bits(value.FloatVal) != math.Float64bits(source.FloatVal) {
			return translationDefect(StageCandidateValidation, "floating candidate changed payload bits")
		}
	}
	return nil
}

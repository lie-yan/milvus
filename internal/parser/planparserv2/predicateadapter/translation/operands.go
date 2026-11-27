package translation

import (
	"math"
	"slices"
	"strconv"

	"github.com/lie-yan/predicate-optimization/expressions"
	"github.com/lie-yan/predicate-optimization/foundation"
	"github.com/lie-yan/predicate-optimization/types"
	valueinterfaces "github.com/lie-yan/predicate-optimization/values/interfaces"

	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
	"github.com/milvus-io/milvus/internal/parser/planparserv2/predicateadapter/schema"
	"github.com/milvus-io/milvus/pkg/v3/proto/planpb"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
	"github.com/milvus-io/milvus/pkg/v3/util/typeutil"
)

// Temporary validated field operand for graph construction.
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
func (t *Translator) readFieldOperand(source *planpb.ColumnInfo) (fieldOperand, limitation, error) {
	field, supported, err := t.view.CheckColumn(source)
	if err != nil {
		return fieldOperand{}, noLimitation, newFailure(UnsafeSource, StageImport, err)
	}
	if !supported {
		return fieldOperand{}, excludedScopeOrVersion, nil
	}
	var typ types.Type
	dataType := field.GetDataType()
	array := dataType == schemapb.DataType_Array
	if array {
		if field.GetElementNullable() || field.GetElementType() == schemapb.DataType_VarChar {
			return fieldOperand{}, excludedDomain, nil
		}
		dataType = field.GetElementType()
	}
	switch dataType {
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
		if field.GetNullable() && !array {
			typ = t.primitives.Int64MayNull
		}
	case schemapb.DataType_Float:
		typ = t.primitives.Float32NonNull
	case schemapb.DataType_Double:
		typ = t.primitives.Float64NonNull
	case schemapb.DataType_VarChar:
		// Match schema parameter lookup without appending into borrowed slice storage.
		params := typeutil.NewKvPairs(slices.Concat(field.GetIndexParams(), field.GetTypeParams()))
		bound, err := params.Get("max_length")
		if err != nil {
			return fieldOperand{}, noLimitation, newFailure(UnsafeSource, StageImport, merr.WrapErrServiceInternalErr(err, "predicate translation: missing VARCHAR bound"))
		}
		maxBytes, err := strconv.ParseUint(bound, 10, 64)
		if err != nil {
			return fieldOperand{}, noLimitation, newFailure(UnsafeSource, StageImport, merr.WrapErrServiceInternalErr(err, "predicate translation: invalid VARCHAR bound"))
		}
		if maxBytes == 0 {
			return fieldOperand{}, noLimitation, sourceDefect("invalid zero VARCHAR schema bound")
		}
		nullability := foundation.NonNull
		if field.GetNullable() {
			nullability = foundation.MayNull
		}
		typ, err = types.NewVarCharType(&maxBytes, nullability)
		if err != nil {
			return fieldOperand{}, noLimitation, newFailure(UnsafeSource, StageImport, err)
		}
	default:
		return fieldOperand{}, excludedDomain, nil
	}
	if array {
		typ, err = types.NewArrayType(typ, foundation.NonNull)
		if err != nil {
			return fieldOperand{}, noLimitation, newFailure(TranslationDefect, StageImport, err)
		}
	}
	if field.GetNullable() && typ.Nullability() != foundation.MayNull {
		typ, err = typ.WithNullability(foundation.MayNull)
		if err != nil {
			return fieldOperand{}, noLimitation, newFailure(TranslationDefect, StageImport, err)
		}
	}
	return fieldOperand{source: source, typ: typ}, noLimitation, nil
}

// Checks actual oneof arms before any default getter or Value construction.
func (t *Translator) readLiteralOperand(source *planpb.GenericValue, typ types.Type) (literalOperand, limitation, error) {
	if source == nil || source.Val == nil {
		return literalOperand{}, noLimitation, sourceDefect("missing literal payload")
	}
	if len(source.ProtoReflect().GetUnknown()) != 0 {
		return literalOperand{}, excludedScopeOrVersion, nil
	}
	if typ == nil {
		return literalOperand{}, noLimitation, sourceDefect("missing literal Type")
	}
	if typ == t.primitives.Int64MayNull {
		typ = t.primitives.Int64NonNull
	} else if typ.Nullability() != foundation.NonNull {
		var err error
		typ, err = typ.WithNullability(foundation.NonNull)
		if err != nil {
			return literalOperand{}, noLimitation, newFailure(TranslationDefect, StageImport, err)
		}
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
	return binding.Reference(), nil
}

// Constructs a scalar graph leaf from an exact native payload.
func (t *Translator) importLiteral(operand literalOperand) (*expressions.LiteralExpression, error) {
	value, err := t.importScalarValue(operand)
	if err != nil {
		return nil, err
	}
	graph, err := expressions.NewLiteralExpression(value)
	if err != nil {
		return nil, newFailure(TranslationDefect, StageImport, err)
	}
	return graph, nil
}

// Copies exact scalar payloads through the selected factory without coercion or evaluation.
func (t *Translator) importScalarValue(operand literalOperand) (valueinterfaces.Value, error) {
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
	return value, nil
}

// Copies canonical metadata from the binding selected by the actual reference.
func exportField(registry *schema.FieldRegistry, graph expressions.Expression) (*planpb.ColumnInfo, error) {
	reference, ok := graph.(*expressions.ReferenceExpression)
	if !ok || reference == nil || registry == nil {
		return nil, translationDefect(StageExport, "expected field reference")
	}
	binding, err := registry.ResolveColumn(reference.ID(), reference.Type())
	if err != nil {
		return nil, newFailure(TranslationDefect, StageExport, err)
	}
	return binding.NewColumnInfo(), nil
}

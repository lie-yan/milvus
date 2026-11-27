package translation

import (
	"github.com/lie-yan/predicate-optimization/expressions"
	"github.com/lie-yan/predicate-optimization/foundation"
	shared "github.com/lie-yan/predicate-optimization/functions"
	"github.com/lie-yan/predicate-optimization/types"
	valueinterfaces "github.com/lie-yan/predicate-optimization/values/interfaces"

	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
	"github.com/milvus-io/milvus/pkg/v3/proto/planpb"
)

// Exposes physical-array containment as a built-in call with typed reference and literal operands.
func (v *importer) importArrayMembership(node *planpb.Expr_JsonContainsExpr) (expressions.Expression, error) {
	n := node.JsonContainsExpr
	if n == nil || n.ColumnInfo == nil {
		return nil, sourceDefect("missing array containment operand")
	}
	if len(n.ProtoReflect().GetUnknown()) != 0 {
		return nil, unsupportedSource(excludedScopeOrVersion)
	}
	if n.ColumnInfo.DataType != schemapb.DataType_Array {
		return nil, unsupportedSource(excludedDomain)
	}
	var name shared.BuiltInFunctionName
	switch n.Op {
	case planpb.JSONContainsExpr_Contains:
		name = shared.BuiltInArrayContains
		if len(n.Elements) != 1 {
			return nil, sourceDefect("contains requires one candidate")
		}
	case planpb.JSONContainsExpr_ContainsAny:
		name = shared.BuiltInArrayContainsAny
	default:
		return nil, unsupportedSource(unsupportedForm)
	}
	if !n.ElementsSameType && len(n.Elements) != 0 {
		return nil, unsupportedSource(excludedDomain)
	}
	owner := v.result.owner
	field, reason, err := owner.readFieldOperand(n.ColumnInfo)
	if err != nil {
		return nil, err
	}
	if reason != noLimitation {
		return nil, unsupportedSource(reason)
	}
	array := field.typ.(*types.ArrayType)
	candidates := make([]valueinterfaces.Value, len(n.Elements))
	for i, value := range n.Elements {
		if err := v.ctx.Err(); err != nil {
			return nil, err
		}
		operand, reason, err := owner.readTermCandidate(value, array.Element())
		if err != nil {
			return nil, err
		}
		if reason != noLimitation {
			return nil, unsupportedSource(reason)
		}
		scalar, err := owner.importScalarValue(operand)
		if err != nil {
			return nil, err
		}
		candidates[i] = scalar
	}
	var candidate valueinterfaces.Value
	if name == shared.BuiltInArrayContains {
		candidate = candidates[0]
	} else {
		candidateType, err := types.NewArrayType(array.Element(), foundation.NonNull)
		if err != nil {
			return nil, newFailure(TranslationDefect, StageImport, err)
		}
		candidate, err = owner.factory.NewArray(candidateType, candidates)
		if err != nil {
			return nil, newFailure(TranslationDefect, StageImport, err)
		}
	}
	literal, err := expressions.NewLiteralExpression(candidate)
	if err != nil {
		return nil, newFailure(TranslationDefect, StageImport, err)
	}
	if err := v.ensureFields(); err != nil {
		return nil, err
	}
	reference, err := importField(v.result.fields, field)
	if err != nil {
		return nil, err
	}
	handle, err := owner.context.LookupBuiltInFunction(name, []types.Type{reference.Type(), literal.Type()})
	if err != nil {
		return nil, newFailure(TranslationDefect, StageImport, err)
	}
	call, err := expressions.NewCallExpression(owner.context, handle, []expressions.Expression{reference, literal})
	if err != nil {
		return nil, newFailure(TranslationDefect, StageImport, err)
	}
	return call, nil
}

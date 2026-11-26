package translation

import (
	"github.com/lie-yan/predicate-optimization/expressions"
	"github.com/lie-yan/predicate-optimization/foundation"
	"github.com/lie-yan/predicate-optimization/types"

	"github.com/milvus-io/milvus/pkg/v3/proto/planpb"
)

// Source-backed target and ordered candidates, independent of logical NOT wrappers.
type termOperands struct {
	// Validated target metadata and full Type.
	field fieldOperand
	// Exact candidate payloads in source order.
	candidates []literalOperand
}

// Admits homogeneous nonempty scalar membership while preserving order and duplicates.
func (v *importer) readTermOperands(node *planpb.Expr_TermExpr) (*termOperands, error) {
	if node == nil || node.TermExpr == nil {
		return nil, sourceDefect("missing term expression")
	}
	n := node.TermExpr
	if n.ColumnInfo == nil {
		return nil, sourceDefect("missing term target")
	}
	if len(n.ProtoReflect().GetUnknown()) != 0 {
		return nil, unsupportedSource(excludedScopeOrVersion)
	}
	if n.IsInField || len(n.Values) == 0 {
		return nil, unsupportedSource(unsupportedForm)
	}
	field, reason, err := v.result.owner.readFieldOperand(n.ColumnInfo)
	if err != nil {
		return nil, err
	}
	if reason != noLimitation {
		return nil, unsupportedSource(reason)
	}
	candidates := make([]literalOperand, len(n.Values))
	for i, value := range n.Values {
		if err := v.ctx.Err(); err != nil {
			return nil, err
		}
		candidate, r, err := v.result.owner.readTermCandidate(value, field.typ)
		if err != nil {
			return nil, err
		}
		if r != noLimitation {
			return nil, unsupportedSource(r)
		}
		candidates[i] = candidate
	}
	return &termOperands{field: field, candidates: candidates}, nil
}

// Classifies heterogeneous/coercion-dependent candidates before invoking exact scalar codecs.
func (t *Translator) readTermCandidate(value *planpb.GenericValue, typ types.Type) (literalOperand, limitation, error) {
	if value == nil || value.Val == nil {
		return literalOperand{}, noLimitation, sourceDefect("missing term candidate")
	}
	if len(value.ProtoReflect().GetUnknown()) != 0 {
		return literalOperand{}, excludedScopeOrVersion, nil
	}
	var matches bool
	switch v := value.Val.(type) {
	case *planpb.GenericValue_BoolVal:
		if v == nil {
			return literalOperand{}, noLimitation, sourceDefect("missing Bool candidate arm")
		}
		_, matches = typ.(*types.BoolType)
	case *planpb.GenericValue_Int64Val:
		if v == nil {
			return literalOperand{}, noLimitation, sourceDefect("missing integer candidate arm")
		}
		_, matches = typ.(*types.IntType)
	case *planpb.GenericValue_FloatVal:
		if v == nil {
			return literalOperand{}, noLimitation, sourceDefect("missing floating candidate arm")
		}
		_, matches = typ.(*types.FloatType)
	case *planpb.GenericValue_StringVal:
		if v == nil {
			return literalOperand{}, noLimitation, sourceDefect("missing string candidate arm")
		}
		_, matches = typ.(*types.VarCharType)
	}
	if !matches {
		return literalOperand{}, excludedDomain, nil
	}
	return t.readLiteralOperand(value, typ)
}

// Builds a dedicated IN node with the target first and literal candidates in their admitted order.
func (v *importer) importTerm(node *planpb.Expr_TermExpr) (expressions.Expression, error) {
	operands, err := v.readTermOperands(node)
	if err != nil {
		return nil, err
	}
	if err := v.ensureFields(); err != nil {
		return nil, err
	}
	field, err := importField(v.result.fields, operands.field)
	if err != nil {
		return nil, err
	}
	owner := v.result.owner
	args := make([]expressions.Expression, len(operands.candidates)+1)
	args[0] = field
	for i, candidate := range operands.candidates {
		if err := v.ctx.Err(); err != nil {
			return nil, err
		}
		args[i+1], err = owner.importLiteral(candidate)
		if err != nil {
			return nil, err
		}
	}
	graph, err := expressions.NewSpecialFormExpression(foundation.SpecialFormIn, owner.primitives.BoolNonNull, args)
	if err != nil {
		return nil, newFailure(TranslationDefect, StageImport, err)
	}
	return graph, nil
}

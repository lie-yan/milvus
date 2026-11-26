package translation

import (
	"github.com/lie-yan/predicate-optimization/expressions"

	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
	"github.com/milvus-io/milvus/internal/parser/planparserv2/predicateadapter/schema"
	translationutils "github.com/milvus-io/milvus/internal/parser/planparserv2/predicateadapter/translation/utils"
	"github.com/milvus-io/milvus/pkg/v3/proto/planpb"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
)

// Checks the complete known O1 payload without constructing operand Types or binding fields.
func checkNullRegion(view *schema.View, source *planpb.Expr, node *planpb.Expr_NullExpr) (limitation, error) {
	if source == nil || node == nil || node.NullExpr == nil || node.NullExpr.ColumnInfo == nil {
		return noLimitation, merr.WrapErrServiceInternalMsg("predicate translation: missing null predicate payload")
	}
	n := node.NullExpr
	if len(source.ProtoReflect().GetUnknown()) != 0 || len(n.ProtoReflect().GetUnknown()) != 0 || len(n.ColumnInfo.ProtoReflect().GetUnknown()) != 0 {
		return excludedScopeOrVersion, nil
	}
	switch n.Op {
	case planpb.NullExpr_Invalid:
		return noLimitation, merr.WrapErrServiceInternalMsg("predicate translation: invalid null operator")
	case planpb.NullExpr_IsNull, planpb.NullExpr_IsNotNull:
	default:
		return unsupportedForm, nil
	}
	field, supported, err := view.CheckColumn(n.ColumnInfo)
	if err != nil {
		return noLimitation, err
	}
	if !supported {
		return excludedScopeOrVersion, nil
	}
	switch field.GetDataType() {
	case schemapb.DataType_Bool, schemapb.DataType_Int8, schemapb.DataType_Int16, schemapb.DataType_Int32, schemapb.DataType_Int64, schemapb.DataType_Float, schemapb.DataType_Double, schemapb.DataType_VarChar:
		return noLimitation, nil
	default:
		return excludedDomain, nil
	}
}

// Registers a complete borrowed region and creates its typed opaque graph leaf.
func (v *importer) importOpaque(source *planpb.Expr, node *planpb.Expr_NullExpr) (expressions.Expression, error) {
	reason, err := checkNullRegion(v.result.owner.view, source, node)
	if err != nil {
		return nil, newFailure(UnsafeSource, StageImport, err)
	}
	if reason != noLimitation {
		return nil, unsupportedSource(reason)
	}
	if v.result.opaques == nil {
		registry, err := translationutils.NewOpaqueRegistry()
		if err != nil {
			return nil, newFailure(TranslationDefect, StageImport, err)
		}
		v.result.opaques = registry
	}
	binding, err := v.result.opaques.Register(source, v.result.owner.primitives.BoolNonNull)
	if err != nil {
		return nil, newFailure(TranslationDefect, StageImport, err)
	}
	graph, err := expressions.NewOpaqueExpression(binding.ID(), binding.Type(), "Milvus null predicate")
	if err != nil {
		return nil, newFailure(TranslationDefect, StageImport, err)
	}
	return graph, nil
}

package translation

import (
	"github.com/lie-yan/predicate-optimization/expressions"
	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
	"github.com/milvus-io/milvus/internal/parser/planparserv2/predicateadapter/schema"
	"github.com/milvus-io/milvus/pkg/v3/proto/planpb"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
	"google.golang.org/protobuf/proto"
)

// Selects O1 only after its source/schema contract and NonNull result are established.
func (a *admitter) admitNull(source *planpb.Expr, node *planpb.Expr_NullExpr) (admissionResult, error) {
	reason, err := checkNullRegion(a.translator.view, source, node)
	if err != nil {
		return admissionResult{}, newFailure(UnsafeSource, StageAdmission, err)
	}
	if reason != noLimitation {
		return admissionResult{limitation: reason}, nil
	}
	id := routeID(len(a.routes))
	a.routes = append(a.routes, admittedRoute{source: source, kind: opaqueNullRoute, typ: a.translator.primitives.BoolNonNull})
	return admissionResult{route: id}, nil
}

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
func (v *importer) importOpaque(route admittedRoute) (occurrenceID, error) {
	id := occurrenceID(len(v.result.occurrences))
	binding, err := v.result.opaques.Register(int(id), route.typ)
	if err != nil {
		return 0, newFailure(TranslationDefect, StageImport, err)
	}
	graph, err := expressions.NewOpaqueExpression(binding.ID(), binding.Type(), "Milvus null predicate")
	if err != nil {
		return 0, newFailure(TranslationDefect, StageImport, err)
	}
	return appendOccurrence(&v.result.occurrences, route.source, graph, nil), nil
}

// Resolves the supplied graph identity rather than substituting the original imported graph.
func resolveOpaque(imported *ImportResult, id occurrenceID, graph *expressions.OpaqueExpression, stage Stage) (*planpb.Expr, error) {
	if id < 0 || int(id) >= len(imported.occurrences) {
		return nil, translationDefect(stage, "missing opaque occurrence")
	}
	record := imported.occurrences[id]
	original, ok := record.imported.(*expressions.OpaqueExpression)
	if !ok || original == nil || graph == nil || len(record.children) != 0 {
		return nil, translationDefect(stage, "opaque occurrence changed shape")
	}
	binding, err := imported.opaques.Resolve(graph.ID(), graph.Type(), int(id))
	if err != nil {
		return nil, newFailure(TranslationDefect, stage, err)
	}
	if !original.ID().EqualsTo(binding.ID()) {
		return nil, translationDefect(stage, "opaque occurrence changed identity")
	}
	return record.source, nil
}

// Copies only the registered region selected by the actual opaque ID and Type.
func (v *exporter) exportOpaque(id occurrenceID, graph *expressions.OpaqueExpression) (*planpb.Expr, error) {
	source, err := resolveOpaque(v.imported, id, graph, StageExport)
	if err != nil {
		return nil, err
	}
	return proto.Clone(source).(*planpb.Expr), nil
}

// Checks O1 eligibility and complete preserved content without invoking the exporter.
func (v *candidateChecker) checkOpaque(record occurrence, candidate *planpb.Expr) error {
	if candidate == nil {
		return translationDefect(StageCandidateValidation, "missing candidate")
	}
	node, ok := candidate.Expr.(*planpb.Expr_NullExpr)
	if !ok {
		return translationDefect(StageCandidateValidation, "candidate changed preserved form")
	}
	reason, err := checkNullRegion(v.imported.admitted.owner.view, candidate, node)
	if err != nil {
		return newFailure(TranslationDefect, StageCandidateValidation, err)
	}
	if reason != noLimitation || !proto.Equal(record.source, candidate) {
		return translationDefect(StageCandidateValidation, "candidate changed preserved content or version")
	}
	return nil
}

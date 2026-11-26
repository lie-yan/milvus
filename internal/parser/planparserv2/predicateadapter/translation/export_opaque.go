package translation

import (
	"github.com/lie-yan/predicate-optimization/expressions"
	"google.golang.org/protobuf/proto"

	"github.com/milvus-io/milvus/pkg/v3/proto/planpb"
)

// Copies only the registered region selected by the actual opaque ID and Type.
func (v *exporter) exportOpaque(graph *expressions.OpaqueExpression) (*planpb.Expr, error) {
	if v.imported.opaques == nil {
		return nil, translationDefect(StageExport, "opaque expression has no preservation registry")
	}
	binding, err := v.imported.opaques.Resolve(graph.ID(), graph.Type())
	if err != nil {
		return nil, newFailure(TranslationDefect, StageExport, err)
	}
	return proto.Clone(binding.Source()).(*planpb.Expr), nil
}

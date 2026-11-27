package translation

import (
	"github.com/lie-yan/predicate-optimization/types"

	"github.com/milvus-io/milvus/internal/parser/planparserv2/predicateadapter/schema"
	"github.com/milvus-io/milvus/pkg/v3/proto/planpb"
)

// Allocates one owning registry at the first admitted field-bearing import.
func (v *importer) ensureFields() error {
	if v.result.fields == nil {
		registry, err := schema.NewFieldRegistry(v.result.owner.view)
		if err != nil {
			return newFailure(TranslationDefect, StageImport, err)
		}
		v.result.fields = registry
	}
	return nil
}

// Shared import/export coverage for exact scalar comparison domains.
func comparisonDomain(typ types.Type, op planpb.OpType) limitation {
	switch typ.(type) {
	case *types.IntType, *types.FloatType:
		return noLimitation
	case *types.BoolType, *types.VarCharType:
		if op == planpb.OpType_Equal || op == planpb.OpType_NotEqual {
			return noLimitation
		}
		return unsupportedForm
	default:
		return excludedDomain
	}
}

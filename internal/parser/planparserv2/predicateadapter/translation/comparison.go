package translation

import (
	"github.com/milvus-io/milvus/internal/parser/planparserv2/predicateadapter/schema"
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

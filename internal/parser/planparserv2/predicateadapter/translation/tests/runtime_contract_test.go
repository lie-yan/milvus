package translation_test

import (
	adapterruntime "github.com/milvus-io/milvus/internal/parser/planparserv2/predicateadapter/runtime"
	"github.com/milvus-io/milvus/internal/parser/planparserv2/predicateadapter/translation"
)

var _ translation.Runtime = (*adapterruntime.Runtime)(nil)

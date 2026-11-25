// Package orchestration owns request inputs and the synchronous predicate translation lifecycle.
package orchestration

import (
	"github.com/milvus-io/milvus/internal/parser/planparserv2/predicateadapter/schema"
	"github.com/milvus-io/milvus/pkg/v3/proto/planpb"
)

// Borrowed per-predicate inputs, stable until the synchronous operation returns.
// Request context and trusted read-only expression validation are separate operation arguments.
type AttemptInput struct {
	// Post-legacy expression after original placement validation; nil does not mean TRUE.
	Original *planpb.Expr
	// Pinned parser schema and top-level membership with immutable field metadata.
	SchemaView *schema.View
}

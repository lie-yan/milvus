// Package utils supplies translation support without depending on stage or mapping state.
package utils

import (
	"strconv"
	"strings"

	expressioninterfaces "github.com/lie-yan/predicate-optimization/expressions/interfaces"
	"github.com/lie-yan/predicate-optimization/types"

	adapterexpressions "github.com/milvus-io/milvus/internal/parser/planparserv2/predicateadapter/expressions"
	"github.com/milvus-io/milvus/internal/parser/planparserv2/predicateadapter/utils"
	"github.com/milvus-io/milvus/pkg/v3/proto/planpb"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
)

// Immutable association between a graph identity and a borrowed source region.
type OpaqueBinding struct {
	// Canonical identity carried by the optimizer graph.
	id *adapterexpressions.OpaqueID
	// Established full result Type, shared read-only.
	typ types.Type
	// Borrowed source region, immutable throughout the import/export attempt.
	source *planpb.Expr
}

// Sequential registration followed by read-only lookup within one import attempt.
type OpaqueRegistry struct {
	// Process-unique preservation scope, separate from field-registry identity.
	scope utils.SessionToken
	// Canonical scope prefix for checked lookup.
	prefix string
	// Ordered bindings indexed by region ordinal, without semantic deduplication.
	bindings []OpaqueBinding
}

// Reserves one preservation scope for an import attempt.
func NewOpaqueRegistry() (*OpaqueRegistry, error) {
	scope, ok := utils.NextSessionToken()
	if !ok {
		return nil, merr.WrapErrServiceInternalMsg("predicateadapter: opaque scopes exhausted")
	}
	return &OpaqueRegistry{scope: scope, prefix: "mvopaque:v1:" + strconv.FormatUint(uint64(scope), 10) + ":"}, nil
}

// Registers a borrowed region and its established immutable result Type.
func (r *OpaqueRegistry) Register(source *planpb.Expr, typ types.Type) (OpaqueBinding, error) {
	if source == nil || typ == nil {
		return OpaqueBinding{}, merr.WrapErrServiceInternalMsg("predicateadapter: missing opaque source or Type")
	}
	binding := OpaqueBinding{id: adapterexpressions.NewOpaqueID(uint64(r.scope), uint64(len(r.bindings))), typ: typ, source: source}
	r.bindings = append(r.bindings, binding)
	return binding, nil
}

// Resolves only registered canonical keys with the expected full Type.
func (r *OpaqueRegistry) Resolve(id expressioninterfaces.OpaqueID, typ types.Type) (OpaqueBinding, error) {
	if id == nil || typ == nil {
		return OpaqueBinding{}, merr.WrapErrServiceInternalMsg("predicateadapter: missing opaque identity or Type")
	}
	key := id.Key()
	suffix, ok := strings.CutPrefix(key, r.prefix)
	if !ok {
		return OpaqueBinding{}, merr.WrapErrServiceInternalMsg("predicateadapter: opaque identity belongs to another scope")
	}
	ordinal, err := strconv.ParseUint(suffix, 10, 64)
	if err != nil {
		// A malformed internal graph identity is a contract defect, not request input.
		return OpaqueBinding{}, merr.WrapErrServiceInternalErr(err, "predicateadapter: invalid opaque ordinal")
	}
	if ordinal >= uint64(len(r.bindings)) {
		return OpaqueBinding{}, merr.WrapErrServiceInternalMsg("predicateadapter: unregistered opaque identity")
	}
	binding := r.bindings[ordinal]
	if binding.id.Key() != key {
		return OpaqueBinding{}, merr.WrapErrServiceInternalMsg("predicateadapter: noncanonical opaque identity")
	}
	if !binding.typ.EqualsTo(typ) {
		return OpaqueBinding{}, merr.WrapErrServiceInternalMsg("predicateadapter: opaque Type disagrees with registration")
	}
	return binding, nil
}

// Canonical identity carried by the preserved graph leaf.
func (b OpaqueBinding) ID() *adapterexpressions.OpaqueID { return b.id }

// Shared immutable result descriptor.
func (b OpaqueBinding) Type() types.Type { return b.typ }

// Borrowed preserved region, which callers must keep read-only.
func (b OpaqueBinding) Source() *planpb.Expr { return b.source }

// Package expressions supplies Milvus-owned expression identities.
package expressions

import (
	"strconv"

	expressioninterfaces "github.com/lie-yan/predicate-optimization/expressions/interfaces"

	"github.com/milvus-io/milvus/internal/parser/planparserv2/predicateadapter/internal/assertions"
)

// Immutable column identity excluding Type, display spelling and source occurrence.
// Schema membership belongs to the registry.
type ColumnID struct {
	key string
}

var _ expressioninterfaces.ColumnID = (*ColumnID)(nil)

// Identifies an original Milvus schema field within an adapter session.
// session must be a nonzero token allocated by the owning adapter.
func NewColumnID(session uint64, fieldID int64) *ColumnID {
	if session == 0 {
		panic("predicateadapter: column identity requires a nonzero session token")
	}
	return &ColumnID{
		key: "mvcol:v1:" + strconv.FormatUint(session, 10) + ":d:" + strconv.FormatInt(fieldID, 10),
	}
}

func (id *ColumnID) Key() string {
	assertions.NotNilPointer(id)
	if id.key == "" {
		panic("predicateadapter: expected constructed column identity")
	}
	return id.key
}

// Accepts any valid implementation of the canonical-key contract,
// including implementations whose Go values cannot be compared with ==.
func (id *ColumnID) EqualsTo(other expressioninterfaces.ColumnID) bool {
	assertions.NotNil(other)
	return id.Key() == other.Key()
}

func (id *ColumnID) String() string { return id.Key() }

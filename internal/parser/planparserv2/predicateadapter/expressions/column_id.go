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
	// Owning adapter session.
	session uint64
	// Original Milvus schema field ID.
	fieldID int64
}

var _ expressioninterfaces.ColumnID = (*ColumnID)(nil)

// Identifies an original Milvus schema field within an adapter session.
// session must be a nonzero token allocated by the owning adapter.
func NewColumnID(session uint64, fieldID int64) *ColumnID {
	if session == 0 {
		panic("predicateadapter: column identity requires a nonzero session token")
	}
	return &ColumnID{session: session, fieldID: fieldID}
}

// Formats the canonical key without storing mutable cache state.
func (id *ColumnID) Key() string {
	id.assertValid()
	return "mvcol:v1:" + strconv.FormatUint(id.session, 10) + ":d:" + strconv.FormatInt(id.fieldID, 10)
}

// Returns the owning adapter session without formatting its key.
func (id *ColumnID) Session() uint64 { id.assertValid(); return id.session }

// Returns the original Milvus field ID without parsing its key.
func (id *ColumnID) FieldID() int64 { id.assertValid(); return id.fieldID }

// Rejects unconstructed identities before exposing their components.
func (id *ColumnID) assertValid() {
	assertions.NotNilPointer(id)
	if id.session == 0 {
		panic("predicateadapter: expected constructed column identity")
	}
}

// Accepts any valid implementation of the canonical-key contract,
// including implementations whose Go values cannot be compared with ==.
func (id *ColumnID) EqualsTo(other expressioninterfaces.ColumnID) bool {
	id.assertValid()
	assertions.NotNil(other)
	if concrete, ok := other.(*ColumnID); ok {
		concrete.assertValid()
		return id.session == concrete.session && id.fieldID == concrete.fieldID
	}
	return id.Key() == other.Key()
}

func (id *ColumnID) String() string { return id.Key() }

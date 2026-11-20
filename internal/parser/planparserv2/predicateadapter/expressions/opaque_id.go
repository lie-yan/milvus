package expressions

import (
	"strconv"

	expressioninterfaces "github.com/lie-yan/predicate-optimization/expressions/interfaces"

	"github.com/milvus-io/milvus/internal/parser/planparserv2/predicateadapter/internal/assertions"
)

// Immutable preservation identity, independent of result Type and diagnostic spelling.
type OpaqueID struct {
	// Canonical attempt scope and region ordinal.
	key string
}

var _ expressioninterfaces.OpaqueID = (*OpaqueID)(nil)

// Identifies one region within a nonzero scope reserved by the owning adapter.
func NewOpaqueID(scope, ordinal uint64) *OpaqueID {
	if scope == 0 {
		panic("predicateadapter: opaque identity requires a nonzero scope")
	}
	return &OpaqueID{key: "mvopaque:v1:" + strconv.FormatUint(scope, 10) + ":" + strconv.FormatUint(ordinal, 10)}
}

// Returns the canonical preservation key; the zero value is invalid.
func (id *OpaqueID) Key() string {
	assertions.NotNilPointer(id)
	if id.key == "" {
		panic("predicateadapter: expected constructed opaque identity")
	}
	return id.key
}

// Accepts compatible identity implementations without requiring Go comparability.
func (id *OpaqueID) EqualsTo(other expressioninterfaces.OpaqueID) bool {
	assertions.NotNil(other)
	return id.Key() == other.Key()
}

// Uses the canonical key for diagnostics without introducing another identity representation.
func (id *OpaqueID) String() string { return id.Key() }

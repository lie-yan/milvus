package functions

import (
	shared "github.com/lie-yan/predicate-optimization/functions"
	functioninterfaces "github.com/lie-yan/predicate-optimization/functions/interfaces"

	"github.com/milvus-io/milvus/internal/parser/planparserv2/predicateadapter/internal/assertions"
)

// Immutable implementation identity within one catalog snapshot.
type functionHandle struct {
	// Owning snapshot, retained for the lifetime of the handle.
	catalog *FunctionCatalog
	// Private canonical entry position, never a serialized identity.
	index int
	// Concrete implementation metadata, shared by fixed handles or materialized during resolution.
	metadata *shared.FunctionMetadata
}

var _ functioninterfaces.FunctionHandle = (*functionHandle)(nil)

// Returns the shared immutable handle for a canonical entry selected by resolution.
func (c *FunctionCatalog) handleAt(index int) *functionHandle {
	h := &c.entries[index].handle
	h.assertValid()
	return h
}

// Returns the registered semantic namespace.
func (h *functionHandle) Namespace() string {
	h.assertValid()
	return h.catalog.entries[h.index].name.Namespace
}

// Returns the registered semantic name, independent of diagnostic labels.
func (h *functionHandle) Name() string {
	h.assertValid()
	return h.catalog.entries[h.index].name.Name
}

// Returns the concrete bound implementation signature.
func (h *functionHandle) Signature() *shared.FunctionSignature {
	h.assertValid()
	return h.metadata.Signature()
}

// Compares owning catalog, canonical declaration and bound signature.
func (h *functionHandle) EqualsTo(other functioninterfaces.FunctionHandle) bool {
	h.assertValid()
	assertions.NotNil(other)
	o, ok := other.(*functionHandle)
	if !ok {
		return false
	}
	o.assertValid()
	return h.catalog == o.catalog && h.index == o.index && (h.metadata == o.metadata || sameSignature(h.Signature(), o.Signature()))
}

// Returns immutable implementation metadata after validating ownership and membership.
func (c *FunctionCatalog) Metadata(handle functioninterfaces.FunctionHandle) (*shared.FunctionMetadata, error) {
	_, err := c.resolveEntry(handle)
	if err != nil {
		return nil, err
	}
	return handle.(*functionHandle).metadata, nil
}

// Distinguishes incompatible representations/snapshots from unknown entries in this snapshot.
func (c *FunctionCatalog) resolveEntry(handle functioninterfaces.FunctionHandle) (*functionEntry, error) {
	assertions.NotNilPointer(c)
	assertions.NotNil(handle)
	h, ok := handle.(*functionHandle)
	if !ok || h.catalog != c {
		return nil, shared.ErrIncompatibleHandle
	}
	if h.index < 0 || h.index >= len(c.entries) {
		return nil, shared.ErrUnknownHandle
	}
	return c.entries[h.index], nil
}

// Enforces the public handle contract on receivers constructed inside this package.
func (h *functionHandle) assertValid() {
	assertions.NotNilPointer(h)
	assertions.NotNilPointer(h.catalog)
	if h.index < 0 || h.index >= len(h.catalog.entries) {
		panic("predicateadapter: expected constructed function handle")
	}
}

// Classifies a recognized handle from operator metadata; foreign or unknown handles return errors.
func (c *FunctionCatalog) IsComparisonFunction(handle functioninterfaces.FunctionHandle) (bool, error) {
	metadata, err := c.Metadata(handle)
	if err != nil {
		return false, err
	}
	op, present := metadata.Operator().Get()
	return present && op.IsComparisonOperator(), nil
}

// Recognizes the arithmetic operator family without inferring semantics from names.
func (c *FunctionCatalog) IsArithmeticFunction(handle functioninterfaces.FunctionHandle) (bool, error) {
	metadata, err := c.Metadata(handle)
	if err != nil {
		return false, err
	}
	op, present := metadata.Operator().Get()
	return present && op.IsArithmeticOperator(), nil
}

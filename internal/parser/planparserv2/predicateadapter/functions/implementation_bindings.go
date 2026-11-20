package functions

import (
	functioninterfaces "github.com/lie-yan/predicate-optimization/functions/interfaces"
	valueinterfaces "github.com/lie-yan/predicate-optimization/values/interfaces"

	"github.com/milvus-io/milvus/internal/parser/planparserv2/predicateadapter/internal/assertions"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
)

// Executes synchronously on immutable implementation-typed Values, returning the implementation result Type without retaining or mutating the argument slice.
type FunctionImplementation func([]valueinterfaces.Value) (valueinterfaces.Value, error)

// Frozen execution bindings sharing the original immutable catalog.
type implementationBindings struct {
	// Owning catalog snapshot used for handle validation.
	catalog *FunctionCatalog
	// Optional execution callbacks keyed by canonical definition identity.
	callbacks map[*functionEntry]FunctionImplementation
}

// Sequential setup owner whose copies share consumption state.
type implementationBindingsBuilder struct {
	// Mutable bindings until ownership transfers to the frozen snapshot.
	state *implementationBindingsState
}

// Shared setup lifetime, preventing copied builders from modifying published bindings.
type implementationBindingsState struct {
	// Catalog and callbacks being assembled.
	bindings implementationBindings
	// Successful freeze has consumed this setup owner.
	built bool
}

// Starts independent execution setup for an existing catalog without copying definitions.
func newImplementationBindingsBuilder(catalog *FunctionCatalog) *implementationBindingsBuilder {
	assertions.NotNilPointer(catalog)
	return &implementationBindingsBuilder{state: &implementationBindingsState{
		bindings: implementationBindings{catalog: catalog, callbacks: make(map[*functionEntry]FunctionImplementation)},
	}}
}

// Associates one non-nil callback with a validated canonical entry, rejecting duplicates without replacement.
func (b *implementationBindingsBuilder) bind(handle functioninterfaces.FunctionHandle, implementation FunctionImplementation) error {
	if err := b.checkOpen(); err != nil {
		return err
	}
	assertions.NotNil(implementation)
	entry, err := b.state.bindings.catalog.resolveEntry(handle)
	if err != nil {
		return err
	}
	if _, exists := b.state.bindings.callbacks[entry]; exists {
		return merr.WrapErrServiceInternalMsg("predicateadapter: duplicate implementation binding for %s.%s", entry.name.Namespace, entry.name.Name)
	}
	b.state.bindings.callbacks[entry] = implementation
	return nil
}

// Transfers private callback storage to an immutable snapshot and consumes every copy of the builder.
func (b *implementationBindingsBuilder) build() (*implementationBindings, error) {
	if err := b.checkOpen(); err != nil {
		return nil, err
	}
	bindings := b.state.bindings
	b.state.bindings = implementationBindings{}
	b.state.built = true
	return &bindings, nil
}

// Rejects unconstructed or consumed owners as internal setup failures.
func (b *implementationBindingsBuilder) checkOpen() error {
	if b == nil || b.state == nil || b.state.built {
		return merr.WrapErrServiceInternalMsg("predicateadapter: implementation bindings builder is uninitialized or consumed")
	}
	return nil
}

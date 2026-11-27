package functions

import (
	"github.com/lie-yan/predicate-optimization/foundation"
	shared "github.com/lie-yan/predicate-optimization/functions"

	"github.com/milvus-io/milvus/pkg/v3/util/merr"
)

// Qualified semantic identity, matched exactly without case normalization.
type FunctionName struct {
	// Semantic namespace, independent of diagnostic labels.
	Namespace string
	// Function name within the namespace.
	Name string
}

// Registration data sharing valid immutable metadata and Type descriptors.
type FunctionDefinition struct {
	// Qualified identity of the declaration.
	Name FunctionName
	// Implementation signature, classifications and null policies.
	Metadata *shared.FunctionMetadata
}

// Immutable canonical definitions; each constructed pointer owns a distinct snapshot.
type FunctionCatalog struct {
	// Private entries in first-registration order.
	entries []*functionEntry
	// Qualified-name overloads.
	byName map[FunctionName]*overloadSet
	// Built-in semantic-identity overloads.
	byBuiltIn map[shared.BuiltInFunctionName]*overloadSet
	// Operator overloads shared by comparison and arithmetic lookup.
	byOperator map[foundation.OperatorType]*overloadSet
}

// Canonical semantic record shared within one snapshot.
type functionEntry struct {
	// Shared fixed-definition handle initialized before catalog publication.
	handle functionHandle
	// Qualified declaration identity.
	name FunctionName
	// Concrete metadata for fixed declarations; nil for parameterized declarations.
	metadata *shared.FunctionMetadata
	// Canonical declaration and execution properties, independent of bound Types.
	declaration FunctionDeclaration
}

// Sequential registration owner; successful Build consumes all copies of the builder.
type FunctionCatalogBuilder struct {
	// Shared lifecycle state, preventing copied builders from bypassing consumption.
	state *builderState
}

// Mutable setup state, inaccessible after publication.
type builderState struct {
	// Canonical entries accumulated before publication.
	entries []*functionEntry
	// Successful publication has consumed this state.
	built bool
}

// Creates an empty registration owner; the zero builder is invalid.
func NewFunctionCatalogBuilder() *FunctionCatalogBuilder {
	return &FunctionCatalogBuilder{state: &builderState{}}
}

// Copies registration data, deduplicating identical semantics and rejecting conflicts without mutation.
func (b *FunctionCatalogBuilder) Register(def FunctionDefinition) error {
	if err := b.checkOpen(); err != nil {
		return err
	}
	if def.Name.Namespace == "" || def.Name.Name == "" || def.Metadata == nil || def.Metadata.Signature() == nil {
		return merr.WrapErrServiceInternalMsg("predicateadapter: invalid function registration")
	}
	return b.register(FunctionDeclaration{
		Name: def.Name, Signature: shared.DeclarationFromSignature(def.Metadata.Signature()),
		DisplayName: def.Metadata.DisplayName(), BuiltInName: def.Metadata.BuiltInName(),
		Operator: def.Metadata.Operator(), NullPolicies: def.Metadata.NullPolicies(),
	}, def.Metadata)
}

// Publishes a complete immutable snapshot and relinquishes the builder's entries.
func (b *FunctionCatalogBuilder) Build() (*FunctionCatalog, error) {
	if err := b.checkOpen(); err != nil {
		return nil, err
	}
	catalog := &FunctionCatalog{entries: b.state.entries,
		byName:     make(map[FunctionName]*overloadSet),
		byBuiltIn:  make(map[shared.BuiltInFunctionName]*overloadSet),
		byOperator: make(map[foundation.OperatorType]*overloadSet),
	}
	for i, entry := range catalog.entries {
		entry.handle = functionHandle{catalog: catalog, index: i, metadata: entry.metadata}
		addDeclaration(catalog.byName, entry.name, i, entry)
		if name, present := entry.declaration.BuiltInName.Get(); present {
			addDeclaration(catalog.byBuiltIn, name, i, entry)
		}
		if op, present := entry.declaration.Operator.Get(); present {
			addDeclaration(catalog.byOperator, op, i, entry)
		}
	}
	b.state.entries = nil
	b.state.built = true
	return catalog, nil
}

// Rejects unconstructed or consumed setup owners as internal setup errors.
func (b *FunctionCatalogBuilder) checkOpen() error {
	if b == nil || b.state == nil || b.state.built {
		return merr.WrapErrServiceInternalMsg("predicateadapter: function catalog builder is uninitialized or consumed")
	}
	return nil
}

// Compares full ordered argument and result Types, including nullability.
func sameSignature(a, b *shared.FunctionSignature) bool {
	if a.ArgumentCount() != b.ArgumentCount() || !a.ResultType().EqualsTo(b.ResultType()) {
		return false
	}
	for i := 0; i < a.ArgumentCount(); i++ {
		if !a.ArgumentTypeAt(i).EqualsTo(b.ArgumentTypeAt(i)) {
			return false
		}
	}
	return true
}

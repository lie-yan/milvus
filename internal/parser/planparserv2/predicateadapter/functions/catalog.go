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
	// Stable implementation identity initialized before catalog publication.
	handle functionHandle
	// Qualified declaration identity.
	name FunctionName
	// Immutable implementation semantics; the first diagnostic label wins.
	metadata *shared.FunctionMetadata
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
	for _, existing := range b.state.entries {
		if existing.name != def.Name || !sameSignature(existing.metadata.Signature(), def.Metadata.Signature()) {
			continue
		}
		if !sameProperties(existing.metadata, def.Metadata) {
			return merr.WrapErrServiceInternalMsg("predicateadapter: conflicting function registration for %s.%s", def.Name.Namespace, def.Name.Name)
		}
		return nil
	}
	b.state.entries = append(b.state.entries, &functionEntry{name: def.Name, metadata: def.Metadata})
	return nil
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
		entry.handle = functionHandle{catalog: catalog, index: i}
		args := entry.metadata.Signature().ArgumentTypes()
		addOverload(catalog.byName, entry.name, i, args)
		if name, present := entry.metadata.BuiltInName().Get(); present {
			addOverload(catalog.byBuiltIn, name, i, args)
		}
		if op, present := entry.metadata.Operator().Get(); present {
			addOverload(catalog.byOperator, op, i, args)
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

// Compares semantic classifications and null policies, excluding diagnostic labels.
func sameProperties(a, b *shared.FunctionMetadata) bool {
	if a.Operator() != b.Operator() || a.BuiltInName() != b.BuiltInName() {
		return false
	}
	for i := 0; i < a.Signature().ArgumentCount(); i++ {
		if a.NullPolicyAt(i) != b.NullPolicyAt(i) {
			return false
		}
	}
	return true
}

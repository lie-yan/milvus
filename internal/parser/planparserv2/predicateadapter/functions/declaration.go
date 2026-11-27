package functions

import (
	"errors"
	"github.com/lie-yan/predicate-optimization/foundation"
	shared "github.com/lie-yan/predicate-optimization/functions"
	"github.com/lie-yan/predicate-optimization/types"

	"github.com/milvus-io/milvus/pkg/v3/util/merr"
)

// Registration syntax and execution properties before argument parameters are bound.
type FunctionDeclaration struct {
	// Qualified semantic identity.
	Name FunctionName
	// Immutable argument and result patterns with declaration-scoped parameters.
	Signature *shared.DeclarationSignature
	// Diagnostic label, excluded from identity.
	DisplayName string
	// Optional framework built-in classification.
	BuiltInName foundation.Optional[shared.BuiltInFunctionName]
	// Optional comparison or arithmetic classification.
	Operator foundation.Optional[foundation.OperatorType]
	// Explicit handling of null arguments, copied on registration.
	NullPolicies []shared.NullPolicy
}

// Registers a declaration without creating placeholder concrete Types or metadata.
func (b *FunctionCatalogBuilder) RegisterDeclaration(def FunctionDeclaration) error {
	if err := b.checkOpen(); err != nil {
		return err
	}
	if def.Name.Namespace == "" || def.Name.Name == "" || def.Signature == nil {
		return merr.WrapErrServiceInternalMsg("predicateadapter: invalid function declaration")
	}
	if name, ok := def.BuiltInName.Get(); ok && name == "" {
		return merr.WrapErrServiceInternalMsg("predicateadapter: empty built-in identity")
	}
	if op, ok := def.Operator.Get(); ok && !op.IsComparisonOperator() && !op.IsArithmeticOperator() {
		return merr.WrapErrServiceInternalMsg("predicateadapter: invalid declaration operator")
	}
	if len(def.NullPolicies) != def.Signature.ArgumentCount() {
		return merr.WrapErrServiceInternalMsg("predicateadapter: declaration null policy count mismatch")
	}
	args := make([]types.Type, def.Signature.ArgumentCount())
	fixed := true
	for i, policy := range def.NullPolicies {
		pattern := def.Signature.ArgumentAt(i)
		if !policy.IsValid() || (policy == shared.PassNull && pattern.Nullability() != foundation.MayNull) {
			return merr.WrapErrServiceInternalMsg("predicateadapter: invalid declaration null policy at %d", i)
		}
		if concrete, ok := pattern.(*types.ConcreteTypePattern); ok {
			args[i] = concrete.Type()
		} else {
			fixed = false
		}
	}
	var metadata *shared.FunctionMetadata
	if fixed {
		var err error
		metadata, err = def.bind(args)
		if err != nil {
			return err
		}
	}
	return b.register(def, metadata)
}

// Canonicalizes equivalent declarations and copies mutable registration storage.
func (b *FunctionCatalogBuilder) register(def FunctionDeclaration, metadata *shared.FunctionMetadata) error {
	for _, entry := range b.state.entries {
		existing := entry.declaration
		if existing.Name != def.Name || !existing.Signature.EqualsTo(def.Signature) {
			continue
		}
		same := existing.Operator == def.Operator && existing.BuiltInName == def.BuiltInName
		for i, policy := range existing.NullPolicies {
			same = same && policy == def.NullPolicies[i]
		}
		if !same {
			return merr.WrapErrServiceInternalMsg("predicateadapter: conflicting function registration for %s.%s", def.Name.Namespace, def.Name.Name)
		}
		return nil
	}
	def.NullPolicies = append([]shared.NullPolicy(nil), def.NullPolicies...)
	b.state.entries = append(b.state.entries, &functionEntry{name: def.Name, metadata: metadata, declaration: def})
	return nil
}

// Materializes concrete implementation semantics before call-site nullability adaptation.
func (d *FunctionDeclaration) bind(args []types.Type) (*shared.FunctionMetadata, error) {
	signature, err := shared.BindDeclarationSignature(d.Signature, args)
	if err != nil {
		return nil, err
	}
	metadata, err := shared.NewFunctionMetadata(signature, d.DisplayName, d.BuiltInName, d.Operator, d.NullPolicies)
	if err != nil {
		return nil, merr.Wrap(errors.Join(shared.ErrContractViolation, err), "predicateadapter: invalid bound declaration metadata")
	}
	return metadata, nil
}

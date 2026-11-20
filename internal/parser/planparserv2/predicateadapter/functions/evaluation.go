package functions

import (
	"errors"

	shared "github.com/lie-yan/predicate-optimization/functions"
	functioninterfaces "github.com/lie-yan/predicate-optimization/functions/interfaces"
	valueinterfaces "github.com/lie-yan/predicate-optimization/values/interfaces"

	"github.com/milvus-io/milvus/internal/parser/planparserv2/predicateadapter/internal/assertions"
	"github.com/milvus-io/milvus/internal/parser/planparserv2/predicateadapter/values"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
)

// Checks the effective call contract, applies null policies and dispatches implementation-typed Values without mutating caller storage.
func (b *implementationBindings) evaluate(handle functioninterfaces.FunctionHandle, signature *shared.FunctionSignature, args []valueinterfaces.Value) (valueinterfaces.Value, error) {
	assertions.NotNilPointer(b)
	assertions.NotNilPointer(signature)
	for _, arg := range args {
		assertions.NotNil(arg)
	}
	entry, err := b.catalog.resolveEntry(handle)
	if err != nil {
		return nil, err
	}
	metadata := entry.metadata
	if err := shared.ValidateCallSignature(metadata, signature); err != nil {
		return nil, err
	}
	if len(args) != signature.ArgumentCount() {
		return nil, merr.Wrap(shared.ErrContractViolation, "predicateadapter: incorrect evaluation argument count")
	}
	for i, arg := range args {
		if !arg.Type().EqualsTo(signature.ArgumentTypeAt(i)) {
			return nil, merr.Wrapf(shared.ErrContractViolation, "predicateadapter: invalid evaluation argument at position %d", i)
		}
	}
	for i, arg := range args {
		if arg.IsNull() && metadata.NullPolicyAt(i) == shared.PropagateNull {
			result, err := values.Factory.NewNull(signature.ResultType())
			if err != nil {
				return nil, merr.Wrap(errors.Join(shared.ErrContractViolation, err), "predicateadapter: null result construction")
			}
			return result, nil
		}
	}
	// Valid Values cannot reach this branch after nullable unspecified-policy binding is rejected.
	for i, arg := range args {
		if arg.IsNull() && metadata.NullPolicyAt(i) == shared.NullPolicyUnspecified {
			return nil, shared.ErrEvaluationDisallowed
		}
	}
	implementation := b.callbacks[entry]
	if implementation == nil {
		return nil, shared.ErrEvaluationUnavailable
	}
	parameters := metadata.Signature()
	adapted := make([]valueinterfaces.Value, len(args))
	for i, arg := range args {
		adapted[i], err = arg.WithNullability(parameters.ArgumentTypeAt(i).Nullability())
		if err != nil {
			return nil, merr.Wrap(errors.Join(shared.ErrContractViolation, err), "predicateadapter: argument adaptation")
		}
	}
	result, err := implementation(adapted)
	if err != nil {
		return nil, classifyEvaluationError(err)
	}
	assertions.NotNil(result)
	if !result.Type().EqualsTo(parameters.ResultType()) {
		return nil, merr.Wrap(shared.ErrContractViolation, "predicateadapter: invalid implementation result Type")
	}
	result, err = result.WithNullability(signature.ResultType().Nullability())
	if err != nil {
		return nil, merr.Wrap(errors.Join(shared.ErrContractViolation, err), "predicateadapter: result adaptation")
	}
	return result, nil
}

// Preserves recognized framework categories and causes, classifying other callback errors as execution failures.
func classifyEvaluationError(err error) error {
	for _, category := range []error{
		shared.ErrUnknownHandle, shared.ErrIncompatibleHandle, shared.ErrUnsupportedLookup,
		shared.ErrEvaluationUnavailable, shared.ErrEvaluationDisallowed,
		shared.ErrEvaluationFailure, shared.ErrContractViolation,
	} {
		if errors.Is(err, category) {
			return err
		}
	}
	return merr.Wrap(errors.Join(shared.ErrEvaluationFailure, err), "predicateadapter: function implementation failed")
}

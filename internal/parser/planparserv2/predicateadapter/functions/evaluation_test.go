package functions

import (
	"errors"
	"testing"

	"github.com/lie-yan/predicate-optimization/foundation"
	shared "github.com/lie-yan/predicate-optimization/functions"
	"github.com/lie-yan/predicate-optimization/types"
	valueinterfaces "github.com/lie-yan/predicate-optimization/values/interfaces"
	"github.com/stretchr/testify/require"

	"github.com/milvus-io/milvus/internal/parser/planparserv2/predicateadapter/values"
)

// Builds one unary definition so execution tests focus on policies rather than operator semantics.
func evaluationCatalog(t *testing.T, policy shared.NullPolicy, parameter types.Type, result types.Type) *FunctionCatalog {
	t.Helper()
	sig, err := shared.NewFunctionSignature([]types.Type{parameter}, result)
	require.NoError(t, err)
	metadata, err := shared.NewFunctionMetadata(sig, "fixture", foundation.Optional[shared.BuiltInFunctionName]{}, foundation.Optional[foundation.OperatorType]{}, []shared.NullPolicy{policy})
	require.NoError(t, err)
	return catalogWith(t, FunctionDefinition{Name: FunctionName{"test", "fixture"}, Metadata: metadata})
}

func evaluationBindings(t *testing.T, catalog *FunctionCatalog, implementation FunctionImplementation) *implementationBindings {
	t.Helper()
	builder := newImplementationBindingsBuilder(catalog)
	if implementation != nil {
		require.NoError(t, builder.bind(catalog.handleAt(0), implementation))
	}
	bindings, err := builder.build()
	require.NoError(t, err)
	return bindings
}

func TestEvaluationNullabilityAndPropagation(t *testing.T) {
	integer, _ := types.NewIntType(types.Int64, foundation.NonNull)
	nullable, _ := types.NewIntType(types.Int64, foundation.MayNull)
	boolean, _ := types.NewBoolType(foundation.NonNull)
	catalog := evaluationCatalog(t, shared.PropagateNull, integer, boolean)
	signature, err := shared.BindFunctionSignature(catalog.entries[0].metadata, []types.Type{nullable})
	require.NoError(t, err)
	input, err := values.Factory.NewInteger(nullable, 9007199254740993)
	require.NoError(t, err)
	expected, err := values.Factory.NewInteger(integer, 9007199254740993)
	require.NoError(t, err)
	calls := 0
	bindings := evaluationBindings(t, catalog, func(args []valueinterfaces.Value) (valueinterfaces.Value, error) {
		calls++
		require.True(t, args[0].Type().EqualsTo(integer))
		require.True(t, args[0].EqualsTo(expected))
		// Deliberately mutate the private argument slice to verify caller storage isolation.
		args[0] = nil
		return values.Factory.NewBool(boolean, true)
	})
	args := []valueinterfaces.Value{input}
	result, err := bindings.evaluate(catalog.handleAt(0), signature, args)
	require.NoError(t, err)
	require.True(t, result.Type().EqualsTo(signature.ResultType()))
	require.False(t, result.IsNull())
	require.True(t, result.(valueinterfaces.BoolValue).Bool())
	require.Same(t, input, args[0])
	require.True(t, input.Type().EqualsTo(nullable))
	null, err := values.Factory.NewNull(nullable)
	require.NoError(t, err)
	for _, candidate := range []*implementationBindings{bindings, evaluationBindings(t, catalog, nil)} {
		result, err := candidate.evaluate(catalog.handleAt(0), signature, []valueinterfaces.Value{null})
		require.NoError(t, err)
		require.True(t, result.IsNull())
		require.True(t, result.Type().EqualsTo(signature.ResultType()))
	}
	require.Equal(t, 1, calls)
	_, err = evaluationBindings(t, catalog, nil).evaluate(catalog.handleAt(0), signature, args)
	require.ErrorIs(t, err, shared.ErrEvaluationUnavailable)
}

func TestEvaluationPassNull(t *testing.T) {
	boolean, _ := types.NewBoolType(foundation.MayNull)
	catalog := evaluationCatalog(t, shared.PassNull, boolean, boolean)
	input, err := values.Factory.NewNull(boolean)
	require.NoError(t, err)
	calls := 0
	bindings := evaluationBindings(t, catalog, func(args []valueinterfaces.Value) (valueinterfaces.Value, error) {
		calls++
		require.True(t, args[0].IsNull())
		return values.Factory.NewBool(boolean, false)
	})
	result, err := bindings.evaluate(catalog.handleAt(0), catalog.entries[0].metadata.Signature(), []valueinterfaces.Value{input})
	require.NoError(t, err)
	require.False(t, result.IsNull())
	require.True(t, result.Type().EqualsTo(boolean))
	require.Equal(t, 1, calls)
}

func TestEvaluationValidationBeforeDispatch(t *testing.T) {
	integer, _ := types.NewIntType(types.Int64, foundation.NonNull)
	narrow, _ := types.NewIntType(types.Int8, foundation.NonNull)
	boolean, _ := types.NewBoolType(foundation.NonNull)
	nullable, _ := types.NewIntType(types.Int64, foundation.MayNull)
	catalog := evaluationCatalog(t, shared.PropagateNull, integer, boolean)
	valid := catalog.entries[0].metadata.Signature()
	input, err := values.Factory.NewInteger(integer, 1)
	require.NoError(t, err)
	null, err := values.Factory.NewNull(nullable)
	require.NoError(t, err)
	widthSignature, _ := shared.NewFunctionSignature([]types.Type{narrow}, boolean)
	resultSignature, _ := shared.NewFunctionSignature([]types.Type{integer}, integer)
	for _, tc := range []struct {
		name      string
		signature *shared.FunctionSignature
		args      []valueinterfaces.Value
	}{
		{"signature width", widthSignature, []valueinterfaces.Value{input}},
		{"signature result", resultSignature, []valueinterfaces.Value{input}},
		{"argument count", valid, nil},
		{"argument Type before null propagation", valid, []valueinterfaces.Value{null}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			for _, implementation := range []FunctionImplementation{nil, func([]valueinterfaces.Value) (valueinterfaces.Value, error) {
				t.Fatal("invalid call reached implementation")
				return nil, nil
			}} {
				result, err := evaluationBindings(t, catalog, implementation).evaluate(catalog.handleAt(0), tc.signature, tc.args)
				require.ErrorIs(t, err, shared.ErrContractViolation)
				require.Nil(t, result)
			}
		})
	}
	// Unspecified NULL handling fails binding, even when the supplied Values are valid.
	unspecified := evaluationCatalog(t, shared.NullPolicyUnspecified, integer, boolean)
	nullableSignature, _ := shared.NewFunctionSignature([]types.Type{nullable}, boolean)
	_, err = evaluationBindings(t, unspecified, nil).evaluate(unspecified.handleAt(0), nullableSignature, []valueinterfaces.Value{null})
	require.ErrorIs(t, err, shared.ErrContractViolation)
	require.ErrorIs(t, err, shared.ErrUnsupportedLookup)
	bindings := evaluationBindings(t, catalog, nil)
	_, err = bindings.evaluate(unspecified.handleAt(0), valid, []valueinterfaces.Value{input})
	require.ErrorIs(t, err, shared.ErrIncompatibleHandle)
	_, err = bindings.evaluate(&functionHandle{catalog: catalog, index: 1}, valid, []valueinterfaces.Value{input})
	require.ErrorIs(t, err, shared.ErrUnknownHandle)
}

// Injects an adaptation failure while retaining the wrapped Value's Type and payload contract.
type adaptationFailure struct {
	valueinterfaces.Value
	cause error
}

func (v adaptationFailure) WithNullability(foundation.Nullability) (valueinterfaces.Value, error) {
	return nil, v.cause
}

func TestEvaluationFailuresPreserveCauses(t *testing.T) {
	boolean, _ := types.NewBoolType(foundation.NonNull)
	integer, _ := types.NewIntType(types.Int64, foundation.NonNull)
	catalog := evaluationCatalog(t, shared.PropagateNull, boolean, boolean)
	value, err := values.Factory.NewBool(boolean, true)
	require.NoError(t, err)
	wrong, err := values.Factory.NewInteger(integer, 0)
	require.NoError(t, err)
	cause := errors.New("injected implementation failure")
	classified := errors.Join(shared.ErrEvaluationUnavailable, cause)
	for _, tc := range []struct {
		name      string
		input     valueinterfaces.Value
		result    valueinterfaces.Value
		err       error
		category  error
		wantCause bool
	}{
		{"argument adaptation", adaptationFailure{value, cause}, value, nil, shared.ErrContractViolation, true},
		{"result adaptation", value, adaptationFailure{value, cause}, nil, shared.ErrContractViolation, true},
		{"wrong result Type", value, wrong, nil, shared.ErrContractViolation, false},
		{"callback error", value, nil, cause, shared.ErrEvaluationFailure, true},
		{"classified callback error", value, nil, classified, shared.ErrEvaluationUnavailable, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			calls := 0
			bindings := evaluationBindings(t, catalog, func([]valueinterfaces.Value) (valueinterfaces.Value, error) {
				calls++
				return tc.result, tc.err
			})
			result, err := bindings.evaluate(catalog.handleAt(0), catalog.entries[0].metadata.Signature(), []valueinterfaces.Value{tc.input})
			require.Nil(t, result)
			require.ErrorIs(t, err, tc.category)
			if tc.wantCause {
				require.ErrorIs(t, err, cause)
			}
			if tc.name == "argument adaptation" {
				require.Zero(t, calls)
			}
			if tc.err == classified {
				require.Same(t, classified, err)
			}
		})
	}
}

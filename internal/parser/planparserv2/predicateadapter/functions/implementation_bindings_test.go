package functions

import (
	"errors"
	"testing"

	"github.com/lie-yan/predicate-optimization/foundation"
	shared "github.com/lie-yan/predicate-optimization/functions"
	functioninterfaces "github.com/lie-yan/predicate-optimization/functions/interfaces"
	valueinterfaces "github.com/lie-yan/predicate-optimization/values/interfaces"
	"github.com/stretchr/testify/require"

	"github.com/milvus-io/milvus/pkg/v3/util/merr"
)

func TestImplementationBindingIdentityAndFailures(t *testing.T) {
	def := registration(t, "less", foundation.LessThan, "", shared.PropagateNull)
	catalog := catalogWith(t, def)
	builder := newImplementationBindingsBuilder(catalog)
	calls := 0
	cause := errors.New("callback result")
	implementation := func([]valueinterfaces.Value) (valueinterfaces.Value, error) {
		calls++
		return nil, cause
	}
	require.NoError(t, builder.bind(catalog.handleAt(0), implementation))
	for _, tc := range []struct {
		name   string
		handle functioninterfaces.FunctionHandle
		want   error
	}{
		{"equivalent handle", catalog.handleAt(0), merr.ErrServiceInternal},
		{"foreign catalog", catalogWith(t, def).handleAt(0), shared.ErrIncompatibleHandle},
		{"unknown entry", &functionHandle{catalog: catalog, index: 1}, shared.ErrUnknownHandle},
	} {
		t.Run(tc.name, func(t *testing.T) {
			require.ErrorIs(t, builder.bind(tc.handle, func([]valueinterfaces.Value) (valueinterfaces.Value, error) {
				t.Fatal("rejected callback was installed")
				return nil, nil
			}), tc.want)
		})
	}
	bindings, err := builder.build()
	require.NoError(t, err)
	require.Zero(t, calls, "setup must not invoke callbacks")
	require.Len(t, bindings.callbacks, 1)
	_, err = bindings.callbacks[catalog.entries[0]](nil)
	require.ErrorIs(t, err, cause)
	require.Equal(t, 1, calls, "failed bindings must preserve the original callback")
}

func TestImplementationBindingOwnershipAndLifecycle(t *testing.T) {
	catalog := catalogWith(t, registration(t, "less", foundation.LessThan, "", shared.PropagateNull))
	builder := newImplementationBindingsBuilder(catalog)
	copyOfBuilder := *builder
	callback := func([]valueinterfaces.Value) (valueinterfaces.Value, error) {
		t.Fatal("setup must not invoke callbacks")
		return nil, nil
	}
	require.NoError(t, builder.bind(catalog.handleAt(0), callback))
	bindings, err := copyOfBuilder.build()
	require.NoError(t, err)
	for _, consumed := range []*implementationBindingsBuilder{builder, &copyOfBuilder} {
		require.ErrorIs(t, consumed.bind(catalog.handleAt(0), callback), merr.ErrServiceInternal)
		_, err := consumed.build()
		require.ErrorIs(t, err, merr.ErrServiceInternal)
	}
	// Separate execution setup may leave the same catalog's entries unbound.
	other, err := newImplementationBindingsBuilder(catalog).build()
	require.NoError(t, err)
	require.Same(t, catalog, bindings.catalog)
	require.Same(t, catalog, other.catalog)
	require.Len(t, bindings.callbacks, 1)
	require.Empty(t, other.callbacks)
}

package runtime

import (
	"sync"
	"sync/atomic"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/milvus-io/milvus/internal/parser/planparserv2/predicateadapter/functions"
	"github.com/milvus-io/milvus/pkg/v3/proto/planpb"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
)

func TestSharedRuntimeConcurrentAcquisition(t *testing.T) {
	const readers = 16
	start := make(chan struct{})
	results := make(chan *Runtime, readers)
	var workers sync.WaitGroup
	for range readers {
		workers.Go(func() {
			<-start
			runtime, err := SharedRuntime()
			if err != nil {
				t.Errorf("acquire runtime: %v", err)
				return
			}
			results <- runtime
		})
	}
	close(start)
	workers.Wait()
	close(results)
	require.Len(t, results, readers)
	var first *Runtime
	for runtime := range results {
		if first == nil {
			first = runtime
		}
		require.Same(t, first, runtime)
		handle, err := runtime.ComparisonFunction(planpb.OpType_Equal, runtime.Types().Int64NonNull, runtime.Types().Int64NonNull)
		require.NoError(t, err)
		metadata, err := first.FunctionContext().Metadata(handle)
		require.NoError(t, err)
		require.Same(t, first.Types().BoolNonNull, metadata.Signature().ResultType())
	}
}

func TestRuntimeInitializationFailureIsRetained(t *testing.T) {
	// Exercise a real declaration failure rather than manufacturing a publication-layer error.
	definitions, err := functions.ComparisonDefinitions(int64NonNullType, boolNonNullType)
	require.NoError(t, err)
	definitions[0].Metadata = nil
	var loader runtimeLoader
	var attempts atomic.Int32
	var setupError error
	build := func() (*Runtime, error) {
		attempts.Add(1)
		runtime, err := assembleRuntime(definitions)
		setupError = err
		return runtime, err
	}
	const readers = 16
	start := make(chan struct{})
	results := make(chan error, readers)
	var workers sync.WaitGroup
	for range readers {
		workers.Go(func() {
			<-start
			runtime, err := loader.load(build)
			if runtime != nil {
				t.Error("failed construction published a runtime")
			}
			results <- err
		})
	}
	close(start)
	workers.Wait()
	close(results)
	require.EqualValues(t, 1, attempts.Load())
	for err := range results {
		require.ErrorIs(t, err, merr.ErrServiceInternal)
		require.Same(t, setupError, err)
	}
	// Recovery requires a new process/owner, not an implicit retry on another request.
	runtime, err := loader.load(func() (*Runtime, error) {
		t.Fatal("cached failure retried construction")
		return NewRuntime()
	})
	require.Nil(t, runtime)
	require.Same(t, setupError, err)
}

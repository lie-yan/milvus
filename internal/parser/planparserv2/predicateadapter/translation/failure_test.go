package translation

import (
	"context"
	"errors"
	"testing"

	shared "github.com/lie-yan/predicate-optimization/functions"
	"github.com/stretchr/testify/require"

	"github.com/milvus-io/milvus/pkg/v3/util/merr"
)

func TestFailureProvenanceAndCause(t *testing.T) {
	cause := merr.WrapErrServiceInternalMsg("schema disagreement")
	for _, tc := range []struct {
		kind  FailureKind
		stage Stage
	}{
		{UnsafeSource, StageAdmission}, {TranslationDefect, StageCandidateValidation},
	} {
		wrapped := merr.Wrap(newFailure(tc.kind, tc.stage, cause), "caller context")
		var failure *Failure
		require.True(t, errors.As(wrapped, &failure))
		require.Equal(t, tc.kind, failure.Kind())
		require.Equal(t, tc.stage, failure.Stage())
		require.ErrorIs(t, wrapped, cause)
		require.ErrorIs(t, wrapped, merr.ErrServiceInternal)
	}
	// Cancellation remains recognizable even if a lower layer carries additional failure provenance.
	cancelled := newFailure(TranslationDefect, StageImport, errors.Join(shared.ErrContractViolation, context.Canceled))
	require.ErrorIs(t, cancelled, context.Canceled)
	require.ErrorIs(t, cancelled, shared.ErrContractViolation)
}

package orchestration

import (
	"context"
	"testing"

	"github.com/stretchr/testify/require"

	adapterruntime "github.com/milvus-io/milvus/internal/parser/planparserv2/predicateadapter/runtime"
	"github.com/milvus-io/milvus/pkg/v3/proto/planpb"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
)

func TestRoundTripGatePrecedesRuntimeAcquisition(t *testing.T) {
	input := gateInput(t)
	for _, missing := range []string{"context", "source", "schema", "validator", "cancelled"} {
		t.Run(missing, func(t *testing.T) {
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			var attemptContext context.Context = ctx
			in := input
			validate := func(*planpb.Expr) error { t.Fatal("invalid attempt reached validator"); return nil }
			var expected error = merr.ErrServiceInternal
			switch missing {
			case "context":
				attemptContext = nil
			case "source":
				in.Original = nil
			case "schema":
				in.SchemaView = nil
			case "validator":
				validate = nil
			case "cancelled":
				cancel()
				expected = context.Canceled
			}
			result, err := roundTrip(attemptContext, in, validate, func() (*adapterruntime.Runtime, error) {
				t.Fatal("invalid attempt loaded runtime")
				return nil, merr.ErrServiceInternal
			}, newTranslator)
			require.True(t, IsAdapterAbort(err))
			require.ErrorIs(t, err, expected)
			require.Equal(t, StageGate, result.Report().Stage)
			require.Nil(t, result.Candidate())
		})
	}
}

func TestRoundTripRuntimeAcquisition(t *testing.T) {
	runtime, err := adapterruntime.NewRuntime()
	require.NoError(t, err)
	cause := merr.WrapErrServiceInternalMsg("runtime initialization failed")
	for _, tc := range []struct {
		name    string
		runtime *adapterruntime.Runtime
		cause   error
		cancel  bool
		outcome Outcome
	}{
		{"success", runtime, nil, false, OutcomeTranslated},
		{"setup failure", nil, cause, false, OutcomeRuntimeUnavailable},
		{"cancel during failed setup", nil, cause, true, OutcomeUnknown},
		{"cancel during successful setup", runtime, nil, true, OutcomeUnknown},
		{"cancellation cause", nil, context.DeadlineExceeded, false, OutcomeUnknown},
		{"missing runtime", nil, nil, false, OutcomeUnknown},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			loads, builds := 0, 0
			result, err := roundTrip(ctx, gateInput(t), func(*planpb.Expr) error { return nil }, func() (*adapterruntime.Runtime, error) {
				loads++
				if tc.cancel {
					cancel()
				}
				return tc.runtime, tc.cause
			}, func(in AttemptInput, got *adapterruntime.Runtime) (pairedTranslator, error) {
				builds++
				require.Same(t, runtime, got)
				return newTranslator(in, got)
			})
			require.Equal(t, 1, loads)
			require.Equal(t, tc.outcome, result.Outcome())
			if tc.outcome == OutcomeTranslated {
				require.NoError(t, err)
				require.Equal(t, 1, builds)
				require.NotNil(t, result.Candidate())
				return
			}
			require.Zero(t, builds)
			require.Nil(t, result.Candidate())
			require.Equal(t, StageSetup, result.Report().Stage)
			if tc.outcome == OutcomeRuntimeUnavailable {
				require.NoError(t, err)
				require.Same(t, cause, result.Report().Cause)
			} else {
				require.True(t, IsAdapterAbort(err))
				switch {
				case tc.cancel:
					require.ErrorIs(t, err, context.Canceled)
				case tc.cause != nil:
					require.ErrorIs(t, err, tc.cause)
				default:
					require.ErrorIs(t, err, merr.ErrServiceInternal)
				}
			}
		})
	}
}

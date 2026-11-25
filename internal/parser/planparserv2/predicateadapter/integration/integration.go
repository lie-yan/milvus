// Package integration assembles adapter inputs and consumes complete attempt outcomes.
package integration

import (
	"context"
	"errors"

	"github.com/milvus-io/milvus/internal/parser/planparserv2/predicateadapter/orchestration"
	adapterschema "github.com/milvus-io/milvus/internal/parser/planparserv2/predicateadapter/schema"
	"github.com/milvus-io/milvus/pkg/v3/proto/planpb"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
	"github.com/milvus-io/milvus/pkg/v3/util/typeutil"
)

// Failure provenance covering local setup and orchestration aborts.
type failure struct{ cause error }

// Adds boundary context without replacing the original error classification.
func (e *failure) Error() string {
	return "predicate optimization: " + e.cause.Error()
}

// Preserves the original code, cause and adapter-abort marker.
func (e *failure) Unwrap() error { return e.cause }

// Recognizes failures from the integration boundary through outer wrapping.
func IsFailure(err error) bool {
	var marked *failure
	return errors.As(err, &marked)
}

// Assembles borrowed inputs and runs the empty pipeline synchronously to completion.
// The required validator must synchronously inspect candidates without mutation or retention.
func Apply(helper *typeutil.SchemaHelper, original *planpb.Expr, validateExpr func(*planpb.Expr) error) (*planpb.Expr, error) {
	if helper == nil {
		return nil, &failure{cause: merr.WrapErrServiceInternalMsg("missing predicate schema helper")}
	}
	view, err := adapterschema.NewView(helper, helper.TopLevelFields())
	if err != nil {
		return nil, &failure{cause: err}
	}
	result, err := orchestration.RoundTrip(context.Background(), orchestration.AttemptInput{
		Original: original, SchemaView: view,
	}, validateExpr)
	if err != nil {
		return nil, &failure{cause: err}
	}
	switch result.Outcome() {
	case orchestration.OutcomeTranslated:
		if result.Candidate() == nil {
			return nil, &failure{cause: merr.WrapErrServiceInternalMsg("predicate adapter accepted a missing candidate")}
		}
		return result.Candidate(), nil
	case orchestration.OutcomeRuntimeUnavailable, orchestration.OutcomeUnsupported, orchestration.OutcomeRetainedAfterFailure:
		return original, nil
	default:
		return nil, &failure{cause: merr.WrapErrServiceInternalMsg("unknown predicate adapter outcome: %d", result.Outcome())}
	}
}

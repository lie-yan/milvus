package orchestration

import (
	"context"

	"github.com/lie-yan/predicate-optimization/concrete_rewriters/array_exclusion_grouping_rewriter"
	"github.com/lie-yan/predicate-optimization/concrete_rewriters/array_membership_normalization_rewriter"
	"github.com/lie-yan/predicate-optimization/concrete_rewriters/boolean_cleanup_rewriter"
	"github.com/lie-yan/predicate-optimization/concrete_rewriters/logical_flattening_rewriter"
	"github.com/lie-yan/predicate-optimization/concrete_rewriters/scoped_contextual_rewriter"
	"github.com/lie-yan/predicate-optimization/expressions"
	"github.com/lie-yan/predicate-optimization/rewriter"

	adapterruntime "github.com/milvus-io/milvus/internal/parser/planparserv2/predicateadapter/runtime"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
)

// Request-local rules use shared immutable metadata without evaluating functions.
func rewriteRules(runtime *adapterruntime.Runtime) []rewriter.ExpressionRewriter[struct{}] {
	return []rewriter.ExpressionRewriter[struct{}]{
		scoped_contextual_rewriter.NewScopedContextualRewriter(runtime.ValueFactory(), runtime.FunctionContext()),
		boolean_cleanup_rewriter.NewBooleanCleanupRewriter(runtime.ValueFactory(), runtime.FunctionContext()),
		logical_flattening_rewriter.NewLogicalFlatteningRewriter(),
		array_membership_normalization_rewriter.NewArrayMembershipNormalizationRewriter(),
		array_exclusion_grouping_rewriter.NewArrayExclusionGroupingRewriter(runtime.ValueFactory()),
	}
}

// Applies the selected sequence before executable export.
func rewrite(ctx context.Context, root expressions.Expression, runtime *adapterruntime.Runtime) (expressions.Expression, error) {
	return applyRewriters(ctx, root, rewriteRules(runtime))
}

// Keeps intermediate immutable graphs private and checks cancellation between passes.
func applyRewriters(ctx context.Context, root expressions.Expression, rules []rewriter.ExpressionRewriter[struct{}]) (expressions.Expression, error) {
	for i, rule := range rules {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		next, err := rewriter.NewExpressionTreeRewriter[struct{}](rule).Rewrite(root, struct{}{})
		if err != nil {
			return nil, merr.Wrapf(err, "predicate rewrite pass %d", i+1)
		}
		root = next
	}
	return root, ctx.Err()
}

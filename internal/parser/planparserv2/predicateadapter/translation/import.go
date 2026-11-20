package translation

import "context"

// Per-import construction state; failed operations never publish its partial tables.
type importer struct {
	// Request context for this invocation.
	ctx context.Context
	// Private graph/registry associations under construction.
	result *ImportResult
}

// Dispatches the admitted route without rerunning eligibility or selecting fallback.
func (v *importer) importExpr(id routeID) (occurrenceID, error) {
	if err := v.ctx.Err(); err != nil {
		return 0, err
	}
	if id < 0 || int(id) >= len(v.result.admitted.routes) {
		return 0, translationDefect(StageImport, "missing admitted route")
	}
	route := v.result.admitted.routes[id]
	switch route.kind {
	case trueRoute, notRoute, andRoute, orRoute:
		return v.importLogical(route)
	case termRoute:
		return v.importTerm(route)
	case binaryRangeRoute:
		return v.importBinaryRange(route)
	case compareRoute:
		return v.importCompare(route)
	case unaryRangeRoute:
		return v.importUnaryRange(route)
	case opaqueNullRoute:
		return v.importOpaque(route)
	default:
		return 0, translationDefect(StageImport, "unimplemented admitted route")
	}
}

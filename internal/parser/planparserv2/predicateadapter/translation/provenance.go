package translation

import (
	"slices"

	"github.com/lie-yan/predicate-optimization/expressions"

	"github.com/milvus-io/milvus/internal/parser/planparserv2/predicateadapter/internal/assertions"
	"github.com/milvus-io/milvus/pkg/v3/proto/planpb"
)

// Position in one import's occurrence table, never a source-pointer or semantic identity.
type occurrenceID int

// One source position paired with an immutable imported root and ordered source-child associations.
type occurrence struct {
	// Borrowed read-only variant, envelope flags and node-specific metadata.
	source *planpb.Expr
	// Original immutable graph used to check content correspondence during reverse traversal.
	imported expressions.Expression
	// Owned ordered indices for source-backed children, not generated range/term graph nodes.
	children []occurrenceID
	// Source-backed field/literal roles for unary comparisons, otherwise nil.
	unary *unaryRangeOperands
	// Ordered field roles for field-to-field comparison, otherwise nil.
	comparison *compareOperands
	// Field and ordered bound roles for a specialized binary range.
	binaryRange *binaryRangeOperands
	// Target and ordered candidate roles for scalar membership.
	term *termOperands
}

// Reports whether the occurrence carries paired field-to-field comparison roles.
func (o occurrence) IsFieldComparison() bool {
	return o.comparison != nil
}

// Reports whether the occurrence carries paired field/literal comparison roles.
func (o occurrence) IsUnaryComparison() bool {
	return o.unary != nil
}

// Reports whether the occurrence owns generated lower/upper comparison roles.
func (o occurrence) IsBinaryRange() bool { return o.binaryRange != nil }

// Reports whether the occurrence owns scalar membership operands.
func (o occurrence) IsTerm() bool { return o.term != nil }

// Appends distinct source occurrences even when source pointers or imported graphs are shared.
func appendOccurrence(table *[]occurrence, source *planpb.Expr, imported expressions.Expression, children []occurrenceID) occurrenceID {
	assertions.NotNilPointer(table)
	assertions.NotNilPointer(source)
	assertions.NotNil(imported)
	id := occurrenceID(len(*table))
	*table = append(*table, occurrence{source: source, imported: imported, children: slices.Clone(children)})
	return id
}

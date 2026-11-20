// Package values supplies the selected Milvus predicate-adapter Value factory.
package values

import (
	bundledvalues "github.com/lie-yan/predicate-optimization/testbed/values"
	valueinterfaces "github.com/lie-yan/predicate-optimization/values/interfaces"
)

// Factory provides stateless, concurrent-safe Value construction without defining translator coverage.
// Do not replace it after initialization.
var Factory valueinterfaces.ValueFactory = bundledvalues.ValueFactory{}

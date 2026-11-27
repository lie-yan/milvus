package touchstone

import "fmt"

// Fixed planning and native count-query workloads on the touchstone schema.
type Workload struct {
	Name  string
	Query string
}

// Balanced disjunctions retain distinct field pairs instead of collapsing to one IN list.
func PerformanceWorkloads() []Workload {
	var disjunction func(int, int) string
	disjunction = func(begin, end int) string {
		if end-begin == 1 {
			return fmt.Sprintf("(pk == %d and brand_id == %d)", begin, begin)
		}
		mid := (begin + end) / 2
		return "(" + disjunction(begin, mid) + " or " + disjunction(mid, end) + ")"
	}
	return []Workload{
		{Name: "small", Query: "pk > 7"},
		{Name: "touchstone", Query: Query},
		{Name: "stable_16", Query: disjunction(0, 16)},
		{Name: "large_128", Query: disjunction(0, 128)},
	}
}

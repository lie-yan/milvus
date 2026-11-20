//go:build !debug

package assertions

// Checks an internal invariant in debug builds; omitted in release builds.
func True(condition bool) {}

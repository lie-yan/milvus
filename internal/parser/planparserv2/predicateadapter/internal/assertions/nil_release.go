//go:build !debug

// Package assertions guards internal programmer preconditions in debug builds.
package assertions

// Omits the check in release builds; the argument is still evaluated.
func NotNil(value any) {}

// Omits the pointer check in release builds; the argument is still evaluated.
func NotNilPointer[T any](value *T) {}

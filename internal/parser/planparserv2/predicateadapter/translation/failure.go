package translation

import (
	"fmt"

	"github.com/milvus-io/milvus/internal/parser/planparserv2/predicateadapter/internal/assertions"
)

// Local failure provenance; D3 decides retention or abort without matching diagnostic strings.
type FailureKind uint8

const (
	// Unsupported source coverage with no successful partial translation.
	UnsupportedSource FailureKind = iota + 1
	// Source/schema contract failure for which forwarding the original is unsafe.
	UnsafeSource
	// Failure in private translation state after successful source admission.
	TranslationDefect
)

// Translation stage where a failure originated, independent of its underlying error category.
type Stage uint8

const (
	// Whole-source checks before graph construction and field binding.
	StageAdmission Stage = iota + 1
	// Checked graph construction from admitted source.
	StageImport
	// Graph validation and fresh protobuf reconstruction from occurrence associations.
	StageExport
	// Independent checks over the completed emitted protobuf.
	StageCandidateValidation
)

// Immutable local classification retaining the original cause without graph/source payloads.
type Failure struct {
	// Coverage, source-integrity or private-candidate distinction.
	kind FailureKind
	// Originating translation stage.
	stage Stage
	// Original typed/sentinel cause, preserved for errors.Is and errors.As.
	cause error
}

// Attaches local stage/provenance without changing the cause's classification.
func newFailure(kind FailureKind, stage Stage, cause error) *Failure {
	assertions.NotNil(cause)
	return &Failure{kind: kind, stage: stage, cause: cause}
}

// Returns the local coverage/integrity classification.
func (f *Failure) Kind() FailureKind { return f.kind }

// Returns the stage that detected the failure.
func (f *Failure) Stage() Stage { return f.stage }

// Exposes the original cause for standard error-chain inspection.
func (f *Failure) Unwrap() error { return f.cause }

// Adds stage and kind to the original cause’s diagnostic text.
func (f *Failure) Error() string {
	return fmt.Sprintf("predicate translation failure (kind=%d, stage=%d): %v", f.kind, f.stage, f.cause)
}

// Package schema validates pinned Milvus schema metadata and binds session-local field references.
package schema

import (
	"slices"

	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
	"github.com/milvus-io/milvus/pkg/v3/proto/planpb"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
	"github.com/milvus-io/milvus/pkg/v3/util/typeutil"
)

// Pinned parser helper and top-level membership from one published schema entry.
// The caller must keep the borrowed field nodes immutable throughout the attempt.
type View struct {
	// Parser's schema helper, including flattened subfield lookup.
	helper *typeutil.SchemaHelper
	// Copied top-level membership with borrowed immutable field nodes.
	topLevel []*schemapb.FieldSchema
}

// Pins a checked helper/membership pair while copying only the membership slice.
func NewView(helper *typeutil.SchemaHelper, topLevel []*schemapb.FieldSchema) (*View, error) {
	if helper == nil {
		return nil, merr.WrapErrServiceInternalMsg("predicateadapter: missing parsing schema helper")
	}
	for _, field := range topLevel {
		if field == nil {
			return nil, merr.WrapErrServiceInternalMsg("predicateadapter: nil top-level schema field")
		}
		resolved, err := helper.GetFieldFromID(field.GetFieldID())
		if err != nil {
			// Relabel ParameterInvalid as an internal error because membership comes from published schema metadata.
			return nil, merr.WrapErrServiceInternalErr(err, "predicateadapter: top-level field absent from parsing schema")
		}
		if resolved != field {
			return nil, merr.WrapErrServiceInternalMsg("predicateadapter: field %d does not belong to the parsing schema view", field.GetFieldID())
		}
	}
	return &View{helper: helper, topLevel: slices.Clone(topLevel)}, nil
}

// Validates top-level metadata without mutation or datatype/nullability restrictions.
// It returns a borrowed field, a coverage miss (nil, false, nil), or an internal inconsistency error; success does not imply translator support.
func (view *View) CheckColumn(info *planpb.ColumnInfo) (*schemapb.FieldSchema, bool, error) {
	if info == nil {
		return nil, false, merr.WrapErrServiceInternalMsg("predicateadapter: missing ColumnInfo")
	}
	// Paths, element scopes and unknown protobuf fields require separate metadata rules.
	if len(info.GetNestedPath()) != 0 || info.GetIsElementLevel() || len(info.ProtoReflect().GetUnknown()) != 0 {
		return nil, false, nil
	}
	field, err := view.helper.GetFieldFromID(info.GetFieldId())
	if err != nil {
		// Relabel the lookup failure as an internal-contract error while retaining its cause.
		return nil, false, merr.WrapErrServiceInternalErr(err, "predicateadapter: column absent from pinned schema")
	}
	if !slices.Contains(view.topLevel, field) {
		return nil, false, nil // Lookup also includes flattened struct-array fields.
	}
	if info.GetDataType() != field.GetDataType() ||
		info.GetNullable() != field.GetNullable() ||
		info.GetElementType() != field.GetElementType() ||
		info.GetIsPrimaryKey() != field.GetIsPrimaryKey() ||
		info.GetIsAutoID() != field.GetAutoID() ||
		info.GetIsPartitionKey() != field.GetIsPartitionKey() ||
		info.GetIsClusteringKey() != field.GetIsClusteringKey() {
		return nil, false, merr.WrapErrServiceInternalMsg("predicateadapter: ColumnInfo disagrees with schema for field %d", info.GetFieldId())
	}
	return field, true, nil
}

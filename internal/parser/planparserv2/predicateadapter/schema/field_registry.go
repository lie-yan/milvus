package schema

import (
	"strconv"
	"strings"

	expressioninterfaces "github.com/lie-yan/predicate-optimization/expressions/interfaces"
	"github.com/lie-yan/predicate-optimization/foundation"
	"github.com/lie-yan/predicate-optimization/types"

	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
	adapterexpressions "github.com/milvus-io/milvus/internal/parser/planparserv2/predicateadapter/expressions"
	"github.com/milvus-io/milvus/internal/parser/planparserv2/predicateadapter/utils"
	"github.com/milvus-io/milvus/pkg/v3/proto/planpb"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
)

// Immutable optimizer identity/Type associated with borrowed read-only Milvus metadata.
// Bindings are returned by value; native export copies canonical column metadata.
type FieldBinding struct {
	// Session-scoped Milvus field identity.
	id *adapterexpressions.ColumnID
	// Full immutable Type derived by the translator.
	typ types.Type
	// Borrowed schema field validated by the pinned view.
	field *schemapb.FieldSchema
	// Canonical top-level metadata created once at registration.
	column *planpb.ColumnInfo
}

// Session-local bindings for sequential import and read-only export lookup.
// Its schema and source metadata must remain unchanged throughout the owning attempt.
type FieldRegistry struct {
	// Pinned schema for column metadata validation.
	view *View
	// Process-unique session identity.
	token utils.SessionToken
	// Canonical document-field key prefix for this session.
	prefix string
	// Registered bindings indexed by original Milvus field ID.
	fields map[int64]FieldBinding
}

// Creates an empty registry with a fresh session identity and the supplied pinned view.
func NewFieldRegistry(view *View) (*FieldRegistry, error) {
	if view == nil {
		return nil, merr.WrapErrServiceInternalMsg("predicateadapter: missing registry schema view")
	}
	token, ok := utils.NextSessionToken()
	if !ok {
		return nil, merr.WrapErrServiceInternalMsg("predicateadapter: session tokens exhausted")
	}
	return &FieldRegistry{
		view: view, token: token,
		prefix: "mvcol:v1:" + strconv.FormatUint(uint64(token), 10) + ":d:",
		fields: make(map[int64]FieldBinding),
	}, nil
}

// Registers checked metadata with the translator-derived immutable Type, or reuses its binding.
// The translator owns datatype conversion; this registry checks nullability and repeated full-Type agreement.
func (r *FieldRegistry) BindColumn(info *planpb.ColumnInfo, typ types.Type) (FieldBinding, bool, error) {
	field, supported, err := r.view.CheckColumn(info)
	if err != nil || !supported {
		return FieldBinding{}, false, err
	}
	if typ == nil {
		return FieldBinding{}, false, merr.WrapErrServiceInternalMsg("predicateadapter: missing reference Type")
	}
	nullable := foundation.NonNull
	if field.GetNullable() {
		nullable = foundation.MayNull
	}
	if typ.Nullability() != nullable {
		return FieldBinding{}, false, merr.WrapErrServiceInternalMsg("predicateadapter: reference nullability disagrees with field %d", field.GetFieldID())
	}
	if existing, ok := r.fields[field.GetFieldID()]; ok {
		if !existing.typ.EqualsTo(typ) {
			return FieldBinding{}, false, merr.WrapErrServiceInternalMsg("predicateadapter: conflicting reference Type for field %d", field.GetFieldID())
		}
		return existing, true, nil
	}
	binding := FieldBinding{
		id:  adapterexpressions.NewColumnID(uint64(r.token), field.GetFieldID()),
		typ: typ, field: field,
		column: &planpb.ColumnInfo{
			FieldId: field.GetFieldID(), DataType: field.GetDataType(),
			IsPrimaryKey: field.GetIsPrimaryKey(), IsAutoID: field.GetAutoID(),
			IsPartitionKey: field.GetIsPartitionKey(), IsClusteringKey: field.GetIsClusteringKey(),
			Nullable: field.GetNullable(), ElementType: field.GetElementType(),
		},
	}
	r.fields[field.GetFieldID()] = binding
	return binding, true, nil
}

// Returns a registered binding only when the canonical session key and full Type match.
// Valid immutable IDs from any compatible implementation are accepted; lookup never registers a field.
func (r *FieldRegistry) ResolveColumn(id expressioninterfaces.ColumnID, typ types.Type) (FieldBinding, error) {
	if id == nil || typ == nil {
		return FieldBinding{}, merr.WrapErrServiceInternalMsg("predicateadapter: missing reference identity or Type")
	}
	key := id.Key()
	suffix, ok := strings.CutPrefix(key, r.prefix)
	if !ok {
		return FieldBinding{}, merr.WrapErrServiceInternalMsg("predicateadapter: reference does not belong to this session")
	}
	fieldID, err := strconv.ParseInt(suffix, 10, 64)
	if err != nil {
		return FieldBinding{}, merr.WrapErrServiceInternalErr(err, "predicateadapter: invalid reference field key")
	}
	binding, ok := r.fields[fieldID]
	if !ok || binding.id.Key() != key {
		return FieldBinding{}, merr.WrapErrServiceInternalMsg("predicateadapter: unregistered reference key")
	}
	if !binding.typ.EqualsTo(typ) {
		return FieldBinding{}, merr.WrapErrServiceInternalMsg("predicateadapter: reference Type disagrees with registered field %d", fieldID)
	}
	return binding, nil
}

// Returns the immutable session-scoped optimizer identity.
func (b FieldBinding) ColumnID() *adapterexpressions.ColumnID { return b.id }

// Returns the shared immutable optimizer Type.
func (b FieldBinding) Type() types.Type { return b.typ }

// Returns the borrowed schema field, which callers must keep read-only.
func (b FieldBinding) Field() *schemapb.FieldSchema { return b.field }

// Creates independently owned output metadata without copying protobuf runtime state.
func (b FieldBinding) NewColumnInfo() *planpb.ColumnInfo {
	c := b.column
	return &planpb.ColumnInfo{
		FieldId: c.FieldId, DataType: c.DataType,
		IsPrimaryKey: c.IsPrimaryKey, IsAutoID: c.IsAutoID,
		IsPartitionKey: c.IsPartitionKey, IsClusteringKey: c.IsClusteringKey,
		Nullable: c.Nullable, ElementType: c.ElementType,
	}
}

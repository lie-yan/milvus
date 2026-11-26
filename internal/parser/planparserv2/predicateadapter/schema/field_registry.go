package schema

import (
	"strconv"
	"strings"

	"github.com/lie-yan/predicate-optimization/expressions"
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
// Bindings are returned by value; native export builds independent column metadata.
type FieldBinding struct {
	// Session-scoped Milvus field identity.
	id *adapterexpressions.ColumnID
	// Shared immutable reference, including the full Type and schema name.
	reference *expressions.ReferenceExpression
	// Borrowed schema field validated by the pinned view.
	field *schemapb.FieldSchema
}

// Session-local bindings for sequential import and read-only export lookup.
// Its schema and source metadata must remain unchanged throughout the owning attempt.
type FieldRegistry struct {
	// Pinned schema for column metadata validation.
	view *View
	// Process-unique session identity.
	token utils.SessionToken
	// Compact bindings in registration order; typical queries reference few fields.
	fields []FieldBinding
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
	if existing, ok := r.lookup(field.GetFieldID()); ok {
		if !existing.Type().EqualsTo(typ) {
			return FieldBinding{}, false, merr.WrapErrServiceInternalMsg("predicateadapter: conflicting reference Type for field %d", field.GetFieldID())
		}
		return existing, true, nil
	}
	binding := FieldBinding{
		id:    adapterexpressions.NewColumnID(uint64(r.token), field.GetFieldID()),
		field: field,
	}
	binding.reference, err = expressions.NewReferenceExpression(binding.id, typ, field.GetName())
	if err != nil {
		return FieldBinding{}, false, err
	}
	r.fields = append(r.fields, binding)
	return binding, true, nil
}

// Searches the small set of registered fields without a separate index.
func (r *FieldRegistry) lookup(fieldID int64) (FieldBinding, bool) {
	for i := range r.fields {
		if r.fields[i].field.GetFieldID() == fieldID {
			return r.fields[i], true
		}
	}
	return FieldBinding{}, false
}

// Resolves concrete numeric identities directly, with canonical-key fallback for compatible implementations.
// Valid immutable IDs from any compatible implementation are accepted; lookup never registers a field.
func (r *FieldRegistry) ResolveColumn(id expressioninterfaces.ColumnID, typ types.Type) (FieldBinding, error) {
	if id == nil || typ == nil {
		return FieldBinding{}, merr.WrapErrServiceInternalMsg("predicateadapter: missing reference identity or Type")
	}
	var fieldID int64
	var key string
	if concrete, ok := id.(*adapterexpressions.ColumnID); ok {
		if concrete.Session() != uint64(r.token) {
			return FieldBinding{}, merr.WrapErrServiceInternalMsg("predicateadapter: reference does not belong to this session")
		}
		fieldID = concrete.FieldID()
	} else {
		key = id.Key()
		suffix, ok := strings.CutPrefix(key, "mvcol:v1:"+strconv.FormatUint(uint64(r.token), 10)+":d:")
		if !ok {
			return FieldBinding{}, merr.WrapErrServiceInternalMsg("predicateadapter: reference does not belong to this session")
		}
		var err error
		fieldID, err = strconv.ParseInt(suffix, 10, 64)
		if err != nil {
			return FieldBinding{}, merr.WrapErrServiceInternalErr(err, "predicateadapter: invalid reference field key")
		}
	}
	binding, ok := r.lookup(fieldID)
	if !ok || (key != "" && binding.id.Key() != key) {
		return FieldBinding{}, merr.WrapErrServiceInternalMsg("predicateadapter: unregistered reference key")
	}
	if !binding.Type().EqualsTo(typ) {
		return FieldBinding{}, merr.WrapErrServiceInternalMsg("predicateadapter: reference Type disagrees with registered field %d", fieldID)
	}
	return binding, nil
}

// Returns the immutable session-scoped optimizer identity.
func (b FieldBinding) ColumnID() *adapterexpressions.ColumnID { return b.id }

// Returns the shared immutable optimizer Type.
func (b FieldBinding) Type() types.Type { return b.reference.Type() }

// Returns the borrowed schema field, which callers must keep read-only.
func (b FieldBinding) Field() *schemapb.FieldSchema { return b.field }

// Creates independently owned output metadata without copying protobuf runtime state.
func (b FieldBinding) NewColumnInfo() *planpb.ColumnInfo {
	c := b.field
	return &planpb.ColumnInfo{
		FieldId: c.GetFieldID(), DataType: c.GetDataType(),
		IsPrimaryKey: c.GetIsPrimaryKey(), IsAutoID: c.GetAutoID(),
		IsPartitionKey: c.GetIsPartitionKey(), IsClusteringKey: c.GetIsClusteringKey(),
		Nullable: c.GetNullable(), ElementType: c.GetElementType(),
	}
}

// Returns the immutable reference shared by occurrences of this session binding.
func (b FieldBinding) Reference() *expressions.ReferenceExpression { return b.reference }

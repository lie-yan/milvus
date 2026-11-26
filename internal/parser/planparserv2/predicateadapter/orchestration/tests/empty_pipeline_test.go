package orchestration_test

import (
	"context"
	"testing"

	"github.com/stretchr/testify/require"
	"google.golang.org/protobuf/proto"

	"github.com/milvus-io/milvus-proto/go-api/v3/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
	"github.com/milvus-io/milvus/internal/parser/planparserv2/predicateadapter/orchestration"
	"github.com/milvus-io/milvus/internal/parser/planparserv2/predicateadapter/schema"
	"github.com/milvus-io/milvus/pkg/v3/proto/planpb"
	"github.com/milvus-io/milvus/pkg/v3/util/typeutil"
)

// Permanent fidelity fixture; pin to an explicit empty configuration before adding rewriters.
func TestEmptyPipelineMixedMappingsAndMetadata(t *testing.T) {
	fields := &schemapb.CollectionSchema{Fields: []*schemapb.FieldSchema{
		{FieldID: 101, Name: "left", DataType: schemapb.DataType_Int16, IsClusteringKey: true},
		{FieldID: 102, Name: "right", DataType: schemapb.DataType_Int16},
		{FieldID: 103, Name: "float", DataType: schemapb.DataType_Float},
		{FieldID: 104, Name: "text", DataType: schemapb.DataType_VarChar, IsPrimaryKey: true, IsPartitionKey: true,
			TypeParams: []*commonpb.KeyValuePair{{Key: "max_length", Value: "20"}}},
		{FieldID: 105, Name: "nullable", DataType: schemapb.DataType_Double, Nullable: true},
	}}
	helper, err := typeutil.CreateSchemaHelper(fields)
	require.NoError(t, err)
	view, err := schema.NewView(helper, fields.Fields)
	require.NoError(t, err)
	columns := make([]*planpb.ColumnInfo, len(fields.Fields))
	for i, field := range fields.Fields {
		columns[i] = &planpb.ColumnInfo{
			FieldId: field.FieldID, DataType: field.DataType, Nullable: field.Nullable,
			IsPrimaryKey: field.IsPrimaryKey, IsPartitionKey: field.IsPartitionKey, IsClusteringKey: field.IsClusteringKey,
		}
	}
	not := func(child *planpb.Expr) *planpb.Expr {
		return &planpb.Expr{IsTemplate: true, Expr: &planpb.Expr_UnaryExpr{UnaryExpr: &planpb.UnaryExpr{Op: planpb.UnaryExpr_Not, Child: child}}}
	}
	binary := func(op planpb.BinaryExpr_BinaryOp, left, right *planpb.Expr) *planpb.Expr {
		return &planpb.Expr{Expr: &planpb.Expr_BinaryExpr{BinaryExpr: &planpb.BinaryExpr{Op: op, Left: left, Right: right}}}
	}
	alwaysTrue := &planpb.Expr{Expr: &planpb.Expr_AlwaysTrueExpr{AlwaysTrueExpr: &planpb.AlwaysTrueExpr{}}}
	comparison := &planpb.Expr{IsTemplate: true, Expr: &planpb.Expr_CompareExpr{CompareExpr: &planpb.CompareExpr{
		LeftColumnInfo: columns[0], RightColumnInfo: columns[1], Op: planpb.OpType_NotEqual,
	}}}
	rangeExpr := &planpb.Expr{IsTemplate: true, Expr: &planpb.Expr_BinaryRangeExpr{BinaryRangeExpr: &planpb.BinaryRangeExpr{
		ColumnInfo: columns[2], LowerInclusive: true, UpperInclusive: false,
		LowerValue:                &planpb.GenericValue{Val: &planpb.GenericValue_FloatVal{FloatVal: 0.5}},
		UpperValue:                &planpb.GenericValue{Val: &planpb.GenericValue_FloatVal{FloatVal: 2}},
		LowerTemplateVariableName: "lower", UpperTemplateVariableName: "upper",
	}}}
	term := &planpb.Expr{IsTemplate: true, Expr: &planpb.Expr_TermExpr{TermExpr: &planpb.TermExpr{
		ColumnInfo: columns[3], TemplateVariableName: "candidates",
		Values: []*planpb.GenericValue{
			{Val: &planpb.GenericValue_StringVal{StringVal: "z"}},
			{Val: &planpb.GenericValue_StringVal{StringVal: "a"}},
			{Val: &planpb.GenericValue_StringVal{StringVal: "z"}},
		},
	}}}
	opaque := &planpb.Expr{Expr: &planpb.Expr_NullExpr{NullExpr: &planpb.NullExpr{
		ColumnInfo: columns[4], Op: planpb.NullExpr_IsNull,
	}}}
	// Right-associated AND makes every specialized form independently inspectable.
	parts := []*planpb.Expr{binary(planpb.BinaryExpr_LogicalOr, alwaysTrue, not(alwaysTrue)), comparison, rangeExpr, not(term)}
	source := opaque
	for i := len(parts) - 1; i >= 0; i-- {
		source = binary(planpb.BinaryExpr_LogicalAnd, parts[i], source)
	}
	before, schemaBefore := proto.Clone(source), proto.Clone(fields)
	input := orchestration.AttemptInput{
		Original: source, SchemaView: view,
	}
	// Explicit executable expectation: no native templates, expanded range, sorted membership.
	expected := proto.Clone(source).(*planpb.Expr)
	expected.GetBinaryExpr().Left.GetBinaryExpr().Right.IsTemplate = false
	expectedCursor := expected.GetBinaryExpr().Right
	expectedCursor.GetBinaryExpr().Left.IsTemplate = false
	expectedCursor = expectedCursor.GetBinaryExpr().Right
	expectedCursor.GetBinaryExpr().Left = binary(planpb.BinaryExpr_LogicalAnd,
		&planpb.Expr{Expr: &planpb.Expr_UnaryRangeExpr{UnaryRangeExpr: &planpb.UnaryRangeExpr{ColumnInfo: columns[2], Op: planpb.OpType_GreaterEqual, Value: rangeExpr.GetBinaryRangeExpr().LowerValue}}},
		&planpb.Expr{Expr: &planpb.Expr_UnaryRangeExpr{UnaryRangeExpr: &planpb.UnaryRangeExpr{ColumnInfo: columns[2], Op: planpb.OpType_LessThan, Value: rangeExpr.GetBinaryRangeExpr().UpperValue}}})
	expectedCursor = expectedCursor.GetBinaryExpr().Right
	expectedCursor.GetBinaryExpr().Left.IsTemplate = false
	expectedTermExpr := expectedCursor.GetBinaryExpr().Left.GetUnaryExpr().Child
	expectedTermExpr.IsTemplate = false
	expectedTermExpr.GetTermExpr().TemplateVariableName = ""
	expectedTermExpr.GetTermExpr().Values[0], expectedTermExpr.GetTermExpr().Values[1] = expectedTermExpr.GetTermExpr().Values[1], expectedTermExpr.GetTermExpr().Values[0]
	placementCalls := 0
	result, err := orchestration.RoundTrip(context.Background(), input, func(candidate *planpb.Expr) error {
		placementCalls++
		require.True(t, proto.Equal(expected, candidate), "placement must receive the complete empty-pipeline output")
		return nil
	})
	require.NoError(t, err)
	require.Equal(t, orchestration.OutcomeTranslated, result.Outcome(), "retention cannot satisfy fidelity")
	require.Equal(t, 1, placementCalls)
	output := result.Candidate()
	require.True(t, proto.Equal(expected, output), "preserve executable values and schema flags")

	// Mutate new mapping payloads and metadata to detect aliases into the borrowed input.
	cursor := output.GetBinaryExpr().Right
	outComparison := cursor.GetBinaryExpr().Left.GetCompareExpr()
	require.NotNil(t, outComparison)
	outComparison.LeftColumnInfo.IsClusteringKey = false
	cursor = cursor.GetBinaryExpr().Right
	outRange := cursor.GetBinaryExpr().Left.GetBinaryExpr().Left.GetUnaryRangeExpr()
	require.NotNil(t, outRange)
	outRange.Value.Val = &planpb.GenericValue_FloatVal{FloatVal: 99}
	outRange.TemplateVariableName = "changed"
	cursor = cursor.GetBinaryExpr().Right
	outTerm := cursor.GetBinaryExpr().Left.GetUnaryExpr().Child.GetTermExpr()
	require.NotNil(t, outTerm)
	outTerm.Values[0].Val = &planpb.GenericValue_StringVal{StringVal: "changed"}
	outTerm.ColumnInfo.IsPrimaryKey = false
	outTerm.Values = append(outTerm.Values, &planpb.GenericValue{Val: &planpb.GenericValue_StringVal{StringVal: "extra"}})
	cursor.GetBinaryExpr().Right.GetNullExpr().ColumnInfo.FieldId = 999
	require.True(t, proto.Equal(before, source))
	require.True(t, proto.Equal(schemaBefore, fields))
}

package translation_test

import (
	"context"
	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
	"github.com/milvus-io/milvus/pkg/v3/proto/planpb"
	"github.com/stretchr/testify/require"
	"google.golang.org/protobuf/proto"
	"testing"
)

func TestMappingCancellation(t *testing.T) {
	translator, null, _ := opaqueFixture(t, schemapb.DataType_Int64, false)
	for name, predicate := range map[string]*planpb.Expr{
		"term":  termSource(null.GetNullExpr().ColumnInfo, integerSource(0), integerSource(1)),
		"unary": unarySource(null.GetNullExpr().ColumnInfo, planpb.OpType_Equal, integerSource(0)),
		"range": rangeSource(null.GetNullExpr().ColumnInfo, integerSource(0), integerSource(1)),
	} {
		t.Run(name, func(t *testing.T) {
			source := binarySource(planpb.BinaryExpr_LogicalAnd, null, predicate)
			imported := importSource(t, translator, source)
			candidate, err := translator.Export(context.Background(), imported, imported.Root())
			require.NoError(t, err)
			for name, stage := range map[string]func(context.Context) error{
				"import": func(ctx context.Context) error {
					result, err := translator.Import(ctx, source)
					if err != nil {
						require.Nil(t, result)
					}
					return err
				},
				"export": func(ctx context.Context) error {
					result, err := translator.Export(ctx, imported, imported.Root())
					if err != nil {
						require.Nil(t, result)
					}
					return err
				},
			} {
				t.Run(name, func(t *testing.T) {
					for checks := 1; checks < 32; checks++ {
						err := stage(&cancellationContext{Context: context.Background(), remaining: checks})
						if err == nil {
							require.Greater(t, checks, 4)
							return
						}
						require.ErrorIs(t, err, context.Canceled)
					}
					t.Fatal("stage did not reach a complete result")
				})
			}
			require.True(t, proto.Equal(expectedExport(source), candidate))

		})
	}
}

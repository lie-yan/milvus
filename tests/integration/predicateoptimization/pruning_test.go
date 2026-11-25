// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
//	http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
package predicateoptimization

import (
	"fmt"
	"time"

	"google.golang.org/protobuf/proto"

	"github.com/milvus-io/milvus-proto/go-api/v3/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v3/milvuspb"
	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
	"github.com/milvus-io/milvus/internal/parser/planparserv2/predicateadapter/orchestration"
	"github.com/milvus-io/milvus/pkg/v3/common"
	"github.com/milvus-io/milvus/pkg/v3/util/funcutil"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
	"github.com/milvus-io/milvus/pkg/v3/util/metric"
	"github.com/milvus-io/milvus/tests/integration"
)

func (s *EmptyPipelineSuite) TestSegmentPruning() {
	ctx := s.Cluster.GetContext()
	name := "fu_clustering_" + funcutil.GenRandomStr()
	dim := 8

	schema := &schemapb.CollectionSchema{
		Name: name,
		Fields: []*schemapb.FieldSchema{
			{FieldID: 100, Name: "pk", DataType: schemapb.DataType_Int64, IsPrimaryKey: true, AutoID: true},
			{FieldID: 101, Name: "cluster_key", DataType: schemapb.DataType_Int64, IsClusteringKey: true},
			{
				FieldID: 102, Name: "vec", DataType: schemapb.DataType_FloatVector,
				TypeParams: []*commonpb.KeyValuePair{{Key: common.DimKey, Value: fmt.Sprint(dim)}},
			},
		},
	}
	marshaled, err := proto.Marshal(schema)
	s.Require().NoError(err)
	status, err := s.Cluster.MilvusClient.CreateCollection(ctx, &milvuspb.CreateCollectionRequest{
		CollectionName: name, Schema: marshaled, ShardsNum: 1,
	})
	s.Require().NoError(err)
	s.Require().Equal(commonpb.ErrorCode_Success, status.GetErrorCode(), status.GetReason())

	// Enough rows that clustering compaction emits more than one segment:
	// pruning can only drop a segment when there is another one to keep.
	const rows = 5000
	for b := 0; b < 20; b++ {
		keys := make([]int64, rows)
		for i := range keys {
			keys[i] = int64((b*rows + i) % 1000)
		}
		insert, err := s.Cluster.MilvusClient.Insert(ctx, &milvuspb.InsertRequest{
			CollectionName: name,
			FieldsData: []*schemapb.FieldData{
				&schemapb.FieldData{FieldName: "cluster_key", Type: schemapb.DataType_Int64, Field: &schemapb.FieldData_Scalars{Scalars: &schemapb.ScalarField{Data: &schemapb.ScalarField_LongData{LongData: &schemapb.LongArray{Data: keys}}}}},
				integration.NewFloatVectorFieldData("vec", rows, dim),
			},
			HashKeys: integration.GenerateHashKeys(rows),
			NumRows:  uint32(rows),
		})
		s.Require().NoError(err)
		s.Require().Equal(commonpb.ErrorCode_Success, insert.GetStatus().GetErrorCode(), insert.GetStatus().GetReason())
		flush, err := s.Cluster.MilvusClient.Flush(ctx, &milvuspb.FlushRequest{CollectionNames: []string{name}})
		s.Require().NoError(err)
		s.WaitForFlush(ctx, flush.GetCollSegIDs()[name].GetData(), flush.GetCollFlushTs()[name], "", name)
	}

	index, err := s.Cluster.MilvusClient.CreateIndex(ctx, &milvuspb.CreateIndexRequest{
		CollectionName: name, FieldName: "vec", IndexName: "_default",
		ExtraParams: integration.ConstructIndexParam(dim, integration.IndexFaissIvfFlat, metric.L2),
	})
	s.Require().NoError(err)
	s.Require().Equal(commonpb.ErrorCode_Success, index.GetErrorCode(), index.GetReason())
	s.WaitForIndexBuilt(ctx, name, "vec")

	compact, err := s.Cluster.MilvusClient.ManualCompaction(ctx, &milvuspb.ManualCompactionRequest{
		CollectionName:  name,
		MajorCompaction: true,
	})
	s.Require().NoError(err)
	s.Require().Equal(commonpb.ErrorCode_Success, compact.GetStatus().GetErrorCode(), compact.GetStatus().GetReason())

	s.Require().Eventually(func() bool {
		state, err := s.Cluster.MilvusClient.GetCompactionState(ctx, &milvuspb.GetCompactionStateRequest{
			CompactionID: compact.GetCompactionID(),
		})
		return err == nil && state.GetState() == commonpb.CompactionState_Completed
	}, 300*time.Second, 2*time.Second, "clustering compaction should finish")

	load, err := s.Cluster.MilvusClient.LoadCollection(ctx, &milvuspb.LoadCollectionRequest{CollectionName: name})
	s.Require().NoError(err)
	s.Require().Equal(commonpb.ErrorCode_Success, load.GetErrorCode(), load.GetReason())
	s.WaitForLoad(ctx, name)

	described, err := s.Cluster.MilvusClient.DescribeCollection(ctx, &milvuspb.DescribeCollectionRequest{CollectionName: name})
	s.Require().NoError(merr.CheckRPCCall(described, err))
	schema = described.GetSchema()
	search := integration.ConstructSearchRequest("", name, "cluster_key == 42", "vec", schemapb.DataType_FloatVector, nil, metric.L2, map[string]any{"nprobe": 4}, 1, dim, 5, -1)
	var baseline *schemapb.SearchResultData
	for _, enabled := range []bool{false, true} {
		s.setEnabled(ctx, enabled)
		for _, expr := range []string{"cluster_key == 42", "cluster_key == -1"} {
			dirs := s.arm(schema, "observe", expr)
			before := s.executionFeatures(ctx)["segment_prune"]
			query, err := s.Cluster.MilvusClient.Query(ctx, &milvuspb.QueryRequest{CollectionName: name, Expr: expr, OutputFields: []string{"count(*)"}, UseDefaultConsistency: true})
			s.Require().NoError(merr.CheckRPCCall(query, err))
			want := int64(100)
			if expr == "cluster_key == -1" {
				want = 0
			}
			s.Equal([]int64{want}, query.GetFieldsData()[0].GetScalars().GetLongData().GetData())
			s.translated(dirs, enabled, orchestration.OutcomeTranslated)
			s.Require().Greater(s.executionFeatures(ctx)["segment_prune"], before, "query must actually prune: %s", expr)
		}
		dirs := s.arm(schema, "observe", search.Dsl)
		before := s.executionFeatures(ctx)["segment_prune"]
		found, err := s.Cluster.MilvusClient.Search(ctx, proto.Clone(search).(*milvuspb.SearchRequest))
		s.Require().NoError(merr.CheckRPCCall(found, err))
		s.translated(dirs, enabled, orchestration.OutcomeTranslated)
		s.Require().Greater(s.executionFeatures(ctx)["segment_prune"], before, "search must actually prune")
		if !enabled {
			baseline = proto.Clone(found.GetResults()).(*schemapb.SearchResultData)
		} else {
			s.Equal(baseline.GetIds(), found.GetResults().GetIds())
			s.Equal(baseline.GetScores(), found.GetResults().GetScores())
		}
	}
}

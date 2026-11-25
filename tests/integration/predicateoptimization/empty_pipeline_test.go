// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package predicateoptimization

import (
	"context"
	"encoding/json"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"
	"google.golang.org/protobuf/proto"

	"github.com/milvus-io/milvus-proto/go-api/v3/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v3/milvuspb"
	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
	"github.com/milvus-io/milvus/internal/parser/planparserv2/predicateadapter/orchestration"
	"github.com/milvus-io/milvus/pkg/v3/util/funcutil"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
	"github.com/milvus-io/milvus/pkg/v3/util/metric"
	"github.com/milvus-io/milvus/tests/integration"
	"github.com/milvus-io/milvus/tests/integration/cluster"
)

// RPC parity for the empty pipeline; future rewriter expectations belong separately.
type EmptyPipelineSuite struct {
	integration.MiniClusterSuite
	controlDir string
}

func (s *EmptyPipelineSuite) SetupSuite() {
	s.controlDir = s.T().TempDir()
	s.WithClusterOptions(cluster.WithMetricsPorts(), cluster.WithProcessEnv(func(role string, _ int64) map[string]string {
		if role == "proxy" {
			return map[string]string{"PREDICATE_TEST_CONTROL": filepath.Join(s.controlDir, "control.json")}
		}
		return nil
	}))
	s.WithMilvusConfig("common.security.featureUsageEnabled", "true")
	s.WithMilvusConfig("common.featureUsage.countersEnabled", "true")
	s.WithMilvusConfig("indexCoord.segment.minSegmentNumRowsToEnableIndex", "64")
	s.WithMilvusConfig("queryNode.enableSegmentPrune", "true")
	s.WithMilvusConfig("dataCoord.segment.maxSize", "4")
	s.WithMilvusConfig("dataCoord.compaction.clustering.preferSegmentSizeRatio", "0.5")
	s.WithMilvusConfig("dataCoord.compaction.clustering.enable", "true")
	s.MiniClusterSuite.SetupSuite()
}

// Wait for the proxy's effective value, rather than assuming config propagation completed.
func (s *EmptyPipelineSuite) setEnabled(ctx context.Context, enabled bool) {
	value := strconv.FormatBool(enabled)
	s.Cluster.MustModifyMilvusConfig(map[string]string{"proxy.enablePredicateOptimization": value})
	client := &http.Client{Timeout: 2 * time.Second}
	s.Require().Eventually(func() bool {
		req, err := http.NewRequestWithContext(ctx, http.MethodGet, "http://"+s.Cluster.DefaultProxy().GetMetricsAddr()+"/api/v1/_cluster/configs", nil)
		if err != nil {
			return false
		}
		resp, err := client.Do(req)
		if err != nil {
			return false
		}
		defer resp.Body.Close()
		var configs map[string]string
		if resp.StatusCode != http.StatusOK || json.NewDecoder(resp.Body).Decode(&configs) != nil {
			return false
		}
		for key, got := range configs {
			if strings.EqualFold(strings.ReplaceAll(key, ".", ""), "proxyenablePredicateOptimization") && strings.HasPrefix(got, value+"[") {
				return true
			}
		}
		return false
	}, 15*time.Second, 100*time.Millisecond)
}

func (s *EmptyPipelineSuite) TestExecutionParity() {
	ctx := s.Cluster.GetContext()
	c := s.Cluster.MilvusClient
	r := s.Require()
	const rows = 2048
	for _, indexed := range []bool{false, true} {
		name := "predicate_empty_" + funcutil.GenRandomStr()
		schema := integration.ConstructSchema(name, 2, false,
			&schemapb.FieldSchema{Name: "pk", DataType: schemapb.DataType_Int64, IsPrimaryKey: true},
			&schemapb.FieldSchema{Name: "i", DataType: schemapb.DataType_Int64, IsPartitionKey: true},
			&schemapb.FieldSchema{Name: "j", DataType: schemapb.DataType_Int64},
			&schemapb.FieldSchema{Name: "b", DataType: schemapb.DataType_Bool},
			&schemapb.FieldSchema{Name: "narrow", DataType: schemapb.DataType_Int32},
			&schemapb.FieldSchema{Name: "f", DataType: schemapb.DataType_Float},
			&schemapb.FieldSchema{Name: "text", DataType: schemapb.DataType_VarChar, TypeParams: []*commonpb.KeyValuePair{{Key: "max_length", Value: "16"}}},
			&schemapb.FieldSchema{Name: "n", DataType: schemapb.DataType_Int64, Nullable: true},
			&schemapb.FieldSchema{Name: "vec", DataType: schemapb.DataType_FloatVector, TypeParams: []*commonpb.KeyValuePair{{Key: "dim", Value: "2"}}},
		)
		encoded, err := proto.Marshal(schema)
		r.NoError(err)
		status, err := c.CreateCollection(ctx, &milvuspb.CreateCollectionRequest{CollectionName: name, Schema: encoded, ShardsNum: 1, NumPartitions: 2})
		r.NoError(merr.CheckRPCCall(status, err))
		flags := make([]bool, rows)
		narrow := make([]int32, rows)
		floats := make([]float32, rows)
		texts := make([]string, rows)
		vectors := make([]float32, 2*rows)
		nullable := integration.NewInt64FieldData("n", rows/2)
		nullable.ValidData = make([]bool, rows)
		for i := 0; i < rows; i++ {
			flags[i] = i%2 == 0
			narrow[i] = int32(i)
			floats[i] = float32(i)
			texts[i] = strconv.Itoa(i % 3)
			vectors[2*i] = float32(i + 1)
			nullable.ValidData[i] = i%2 == 0
		}
		fields := []*schemapb.FieldData{integration.NewInt64FieldData("pk", rows), integration.NewInt64FieldData("i", rows), integration.NewInt64FieldDataWithStart("j", rows, 1), nullable,
			{FieldName: "narrow", Type: schemapb.DataType_Int32, Field: &schemapb.FieldData_Scalars{Scalars: &schemapb.ScalarField{Data: &schemapb.ScalarField_IntData{IntData: &schemapb.IntArray{Data: narrow}}}}},
			{FieldName: "f", Type: schemapb.DataType_Float, Field: &schemapb.FieldData_Scalars{Scalars: &schemapb.ScalarField{Data: &schemapb.ScalarField_FloatData{FloatData: &schemapb.FloatArray{Data: floats}}}}},
			{FieldName: "text", Type: schemapb.DataType_VarChar, Field: &schemapb.FieldData_Scalars{Scalars: &schemapb.ScalarField{Data: &schemapb.ScalarField_StringData{StringData: &schemapb.StringArray{Data: texts}}}}},
			{FieldName: "b", Type: schemapb.DataType_Bool, Field: &schemapb.FieldData_Scalars{Scalars: &schemapb.ScalarField{Data: &schemapb.ScalarField_BoolData{BoolData: &schemapb.BoolArray{Data: flags}}}}},
			{FieldName: "vec", Type: schemapb.DataType_FloatVector, Field: &schemapb.FieldData_Vectors{Vectors: &schemapb.VectorField{Dim: 2, Data: &schemapb.VectorField_FloatVector{FloatVector: &schemapb.FloatArray{Data: vectors}}}}},
		}
		inserted, err := c.Insert(ctx, &milvuspb.InsertRequest{CollectionName: name, FieldsData: fields, NumRows: rows, HashKeys: integration.GenerateHashKeys(rows)})
		r.NoError(merr.CheckRPCCall(inserted, err))
		flushed, err := c.Flush(ctx, &milvuspb.FlushRequest{CollectionNames: []string{name}})
		r.NoError(merr.CheckRPCCall(flushed, err))
		s.WaitForFlush(ctx, flushed.GetFlushCollSegIDs()[name].GetData(), flushed.GetCollFlushTs()[name], "", name)
		status, err = c.CreateIndex(ctx, &milvuspb.CreateIndexRequest{CollectionName: name, FieldName: "vec", ExtraParams: integration.ConstructIndexParam(2, integration.IndexFaissIDMap, metric.L2)})
		r.NoError(merr.CheckRPCCall(status, err))
		s.WaitForIndexBuilt(ctx, name, "vec")
		if indexed {
			status, err = c.CreateIndex(ctx, &milvuspb.CreateIndexRequest{CollectionName: name, FieldName: "i", ExtraParams: []*commonpb.KeyValuePair{{Key: "index_type", Value: "INVERTED"}}})
			r.NoError(merr.CheckRPCCall(status, err))
			s.WaitForIndexBuilt(ctx, name, "i")
		}
		status, err = c.LoadCollection(ctx, &milvuspb.LoadCollectionRequest{CollectionName: name})
		r.NoError(merr.CheckRPCCall(status, err))
		s.WaitForLoad(ctx, name)
		described, err := c.DescribeCollection(ctx, &milvuspb.DescribeCollectionRequest{CollectionName: name})
		r.NoError(merr.CheckRPCCall(described, err))
		schema = described.GetSchema()
		cases := []struct {
			expr  string
			match func(int) bool
		}{
			{"i > 7", func(i int) bool { return i > 7 }},
			{"i == 8", func(i int) bool { return i == 8 }},
			{"b != false", func(i int) bool { return i%2 == 0 }},
			{"3 <= i < 12", func(i int) bool { return i >= 3 && i < 12 }},
			{"i < j", func(int) bool { return true }},
			{"i in [9, 2, 9, 0]", func(i int) bool { return i == 9 || i == 2 || i == 0 }},
			{"i > 7 and n is null", func(i int) bool { return i > 7 && i%2 != 0 }},
			{"i + 1 < 20", func(i int) bool { return i < 19 }},
			{"3 <= narrow < 12", func(i int) bool { return i >= 3 && i < 12 }},
			{"3.0 <= f < 12.0", func(i int) bool { return i >= 3 && i < 12 }},
			{`text in ["0", "2"]`, func(i int) bool { return i%3 != 1 }},
		}
		// One request supplies identical query vectors to both switch modes.
		search := integration.ConstructSearchRequest("", name, "i > 7", "vec", schemapb.DataType_FloatVector, nil, metric.L2, map[string]any{}, 1, 2, 5, -1)
		placeholders, err := proto.Marshal(&commonpb.PlaceholderGroup{Placeholders: []*commonpb.PlaceholderValue{{Tag: "$0", Type: commonpb.PlaceholderType_FloatVector, Values: [][]byte{make([]byte, 8)}}}})
		r.NoError(err)
		search.SearchInput = &milvuspb.SearchRequest_PlaceholderGroup{PlaceholderGroup: placeholders}
		var baselineSearch *schemapb.SearchResultData
		var baselineHybrid *schemapb.SearchResultData
		scorerBaselines := map[string]*schemapb.SearchResultData{}
		for _, enabled := range []bool{false, true} {
			s.setEnabled(ctx, enabled)
			for _, tc := range cases {
				var expected []int64
				for i := 0; i < rows; i++ {
					if tc.match(i) {
						expected = append(expected, int64(i))
					}
				}
				before := s.executionFeatures(ctx)
				dirs := s.arm(schema, "observe", tc.expr)
				result, err := c.Query(ctx, &milvuspb.QueryRequest{CollectionName: name, Expr: tc.expr, OutputFields: []string{"pk"}, ConsistencyLevel: commonpb.ConsistencyLevel_Strong})
				r.NoError(merr.CheckRPCCall(result, err), "indexed=%v enabled=%v expr=%s", indexed, enabled, tc.expr)
				outcome := orchestration.OutcomeTranslated
				if tc.expr == "i + 1 < 20" {
					outcome = orchestration.OutcomeUnsupported
				}
				s.translated(dirs, enabled, outcome)
				if tc.expr == "i == 8" {
					after := s.executionFeatures(ctx)
					feature := "filter_exec_path=brute_force"
					if indexed {
						feature = "scalar_index_type=INVERTED"
					}
					r.Greater(after[feature], before[feature], "indexed=%v enabled=%v", indexed, enabled)
				}
				got := append([]int64(nil), result.GetFieldsData()[0].GetScalars().GetLongData().GetData()...)
				sort.Slice(got, func(i, j int) bool { return got[i] < got[j] })
				r.Equal(expected, got, tc.expr)
				dirs = s.arm(schema, "observe", tc.expr)
				count, err := c.Query(ctx, &milvuspb.QueryRequest{CollectionName: name, Expr: tc.expr, OutputFields: []string{"count(*)"}, ConsistencyLevel: commonpb.ConsistencyLevel_Strong})
				r.NoError(merr.CheckRPCCall(count, err))
				s.translated(dirs, enabled, outcome)
				r.Equal(int64(len(expected)), count.GetFieldsData()[0].GetScalars().GetLongData().GetData()[0])
			}
			dirs := s.arm(schema, "observe", search.Dsl)
			found, err := c.Search(ctx, proto.Clone(search).(*milvuspb.SearchRequest))
			r.NoError(merr.CheckRPCCall(found, err))
			s.translated(dirs, enabled, orchestration.OutcomeTranslated)
			r.Equal([]int64{8, 9, 10, 11, 12}, found.GetResults().GetIds().GetIntId().GetData())
			if !enabled {
				baselineSearch = proto.Clone(found.GetResults()).(*schemapb.SearchResultData)
			} else {
				r.Equal(baselineSearch.GetIds(), found.GetResults().GetIds())
				r.Equal(baselineSearch.GetScores(), found.GetResults().GetScores())
			}
			for _, pair := range []struct{ main, scorer string }{{"i > 7", "i < 12"}, {"i + 1 < 20", "i < 12"}, {"i > 7", "i + 1 < 20"}, {"", "i < 12"}} {
				request := proto.Clone(search).(*milvuspb.SearchRequest)
				request.Dsl = pair.main
				request.FunctionScore = &schemapb.FunctionScore{Functions: []*schemapb.FunctionSchema{{Name: "boost", Type: schemapb.FunctionType_Rerank, Params: []*commonpb.KeyValuePair{{Key: "reranker", Value: "boost"}, {Key: "filter", Value: pair.scorer}, {Key: "weight", Value: "2"}}}}}
				dirs = s.arm(schema, "observe", pair.main, pair.scorer)
				result, err := c.Search(ctx, request)
				r.NoError(merr.CheckRPCCall(result, err))
				var outcomes []orchestration.Outcome
				for _, source := range []string{pair.main, pair.scorer} {
					if source == "" {
						continue
					}
					outcome := orchestration.OutcomeTranslated
					if strings.Contains(source, "+") {
						outcome = orchestration.OutcomeUnsupported
					}
					outcomes = append(outcomes, outcome)
				}
				s.translated(dirs, enabled, outcomes...)
				key := pair.main + "/" + pair.scorer
				if !enabled {
					scorerBaselines[key] = proto.Clone(result.GetResults()).(*schemapb.SearchResultData)
				} else {
					r.Equal(scorerBaselines[key].GetIds(), result.GetResults().GetIds(), key)
					r.Equal(scorerBaselines[key].GetScores(), result.GetResults().GetScores(), key)
				}
			}
			templated, err := c.Query(ctx, &milvuspb.QueryRequest{CollectionName: name, Expr: "i > {bound}", ExprTemplateValues: map[string]*schemapb.TemplateValue{"bound": {Val: &schemapb.TemplateValue_Int64Val{Int64Val: 7}}}, OutputFields: []string{"count(*)"}})
			r.NoError(merr.CheckRPCCall(templated, err))
			r.Equal(int64(rows-8), templated.GetFieldsData()[0].GetScalars().GetLongData().GetData()[0])
			retained := proto.Clone(search).(*milvuspb.SearchRequest)
			retained.Dsl = "i + 1 > 8"
			dirs = s.arm(schema, "observe", search.Dsl, retained.Dsl)
			hybrid, err := c.HybridSearch(ctx, &milvuspb.HybridSearchRequest{CollectionName: name, Requests: []*milvuspb.SearchRequest{proto.Clone(search).(*milvuspb.SearchRequest), retained}, RankParams: []*commonpb.KeyValuePair{{Key: "strategy", Value: "rrf"}, {Key: "params", Value: `{"k":60}`}, {Key: "limit", Value: "5"}}})
			r.NoError(merr.CheckRPCCall(hybrid, err))
			s.translated(dirs, enabled, orchestration.OutcomeTranslated, orchestration.OutcomeUnsupported)
			r.Equal([]int64{8, 9, 10, 11, 12}, hybrid.GetResults().GetIds().GetIntId().GetData())
			if !enabled {
				baselineHybrid = proto.Clone(hybrid.GetResults()).(*schemapb.SearchResultData)
			} else {
				r.Equal(baselineHybrid.GetIds(), hybrid.GetResults().GetIds())
				r.Equal(baselineHybrid.GetScores(), hybrid.GetResults().GetScores())
			}
			invalid, err := c.Query(ctx, &milvuspb.QueryRequest{CollectionName: name, Expr: "i >", OutputFields: []string{"pk"}})
			r.NoError(err)
			r.Equal(int32(1100), invalid.GetStatus().GetCode())
		}
		if !indexed {
			s.verifyFailures(ctx, name, schema, search)
		}
		r.NoError(os.Remove(filepath.Join(s.controlDir, "control.json")))
		status, err = c.ReleaseCollection(ctx, &milvuspb.ReleaseCollectionRequest{CollectionName: name})
		r.NoError(merr.CheckRPCCall(status, err))
		status, err = c.DropCollection(ctx, &milvuspb.DropCollectionRequest{CollectionName: name})
		r.NoError(merr.CheckRPCCall(status, err))
	}
}

func TestEmptyPipeline(t *testing.T) { suite.Run(t, new(EmptyPipelineSuite)) }

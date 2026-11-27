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
	"os"
	"path/filepath"
	"sort"
	"strconv"

	"google.golang.org/protobuf/proto"

	"github.com/milvus-io/milvus-proto/go-api/v3/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v3/milvuspb"
	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
	"github.com/milvus-io/milvus/internal/parser/planparserv2/predicateadapter/orchestration"
	"github.com/milvus-io/milvus/internal/parser/planparserv2/predicateadapter/tests/touchstone"
	"github.com/milvus-io/milvus/pkg/v3/util/funcutil"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
	"github.com/milvus-io/milvus/pkg/v3/util/metric"
	"github.com/milvus-io/milvus/tests/integration"
)

// Queries P and NOT P separately so UNKNOWN cannot masquerade as FALSE.
func (s *EmptyPipelineSuite) TestTouchstoneExecutionParity() { s.touchstoneExecutionParity(false) }

func (s *EmptyPipelineSuite) TestTouchstoneOptimizedExecutionParity() {
	s.touchstoneExecutionParity(true)
}

func (s *EmptyPipelineSuite) touchstoneExecutionParity(optimized bool) {
	ctx, c, r := s.Cluster.GetContext(), s.Cluster.MilvusClient, s.Require()
	for _, nullableArticle := range []bool{false, true} {
		name := "predicate_touchstone_" + funcutil.GenRandomStr()
		schema := touchstone.Schema(name, nullableArticle)
		encoded, err := proto.Marshal(schema)
		r.NoError(err)
		status, err := c.CreateCollection(ctx, &milvuspb.CreateCollectionRequest{CollectionName: name, Schema: encoded, ShardsNum: 1})
		r.NoError(merr.CheckRPCCall(status, err))
		cases, err := touchstone.Cases(nullableArticle)
		r.NoError(err)
		// Four copies exceed the suite's 64-row indexing threshold without changing the cases.
		var rows []touchstone.Row
		expected := map[string][]int64{"TRUE": {}, "FALSE": {}, "UNKNOWN": {}, "array": {}}
		for copy := int64(0); copy < 4; copy++ {
			for _, tc := range cases {
				row := tc.Row
				row.PK += copy * 1000
				rows = append(rows, row)
				expected[tc.Expected] = append(expected[tc.Expected], row.PK)
				for _, tag := range row.Tags {
					if tag == 8791004 {
						expected["array"] = append(expected["array"], row.PK)
						break
					}
				}
			}
		}
		inserted, err := c.Insert(ctx, &milvuspb.InsertRequest{CollectionName: name, FieldsData: touchstoneFields(rows, nullableArticle), NumRows: uint32(len(rows)), HashKeys: integration.GenerateHashKeys(len(rows))})
		r.NoError(merr.CheckRPCCall(inserted, err))
		flushed, err := c.Flush(ctx, &milvuspb.FlushRequest{CollectionNames: []string{name}})
		r.NoError(merr.CheckRPCCall(flushed, err))
		s.WaitForFlush(ctx, flushed.GetFlushCollSegIDs()[name].GetData(), flushed.GetCollFlushTs()[name], "", name)
		status, err = c.CreateIndex(ctx, &milvuspb.CreateIndexRequest{CollectionName: name, FieldName: "vector", ExtraParams: integration.ConstructIndexParam(2, integration.IndexFaissIDMap, metric.L2)})
		r.NoError(merr.CheckRPCCall(status, err))
		s.WaitForIndexBuilt(ctx, name, "vector")
		described, err := c.DescribeCollection(ctx, &milvuspb.DescribeCollectionRequest{CollectionName: name})
		r.NoError(merr.CheckRPCCall(described, err))
		schema = described.GetSchema()
		for _, indexed := range []bool{false, true} {
			if indexed {
				for _, field := range []string{"article_type", "is_review", "brand_id", "tag_ids", "category_ids", "mall_ids"} {
					status, err = c.CreateIndex(ctx, &milvuspb.CreateIndexRequest{CollectionName: name, FieldName: field, ExtraParams: []*commonpb.KeyValuePair{{Key: "index_type", Value: "INVERTED"}}})
					r.NoError(merr.CheckRPCCall(status, err))
					s.WaitForIndexBuilt(ctx, name, field)
				}
			}
			status, err = c.LoadCollection(ctx, &milvuspb.LoadCollectionRequest{CollectionName: name})
			r.NoError(merr.CheckRPCCall(status, err))
			s.WaitForLoad(ctx, name)
			actions := []string{"observe"}
			if optimized {
				actions = []string{"optimize:1", "optimize:2", "optimize:3", "optimize:4", "optimize:5"}
			}
			for _, action := range actions {
				for _, enabled := range []bool{false, true} {
					s.T().Logf("action=%s touchstone nullable_article=%v indexed=%v enabled=%v rows=%d", action, nullableArticle, indexed, enabled, len(rows))
					s.setEnabled(ctx, enabled)
					positive := s.touchstoneQuery(ctx, name, schema, touchstone.Query, enabled, action)
					negative := s.touchstoneQuery(ctx, name, schema, "not ("+touchstone.Query+")", enabled, action)
					r.Equal(expected["TRUE"], positive)
					r.Equal(expected["FALSE"], negative)
					classified := map[int64]bool{}
					for _, id := range positive {
						classified[id] = true
					}
					for _, id := range negative {
						r.False(classified[id], "row selected by both P and NOT P")
						classified[id] = true
					}
					var unknown []int64
					for _, row := range rows {
						if !classified[row.PK] {
							unknown = append(unknown, row.PK)
						}
					}
					r.Equal(expected["UNKNOWN"], unknown)
					// Confirm the intended native array path, independently of full-query short circuiting.
					before := s.executionFeatures(ctx)
					got := s.touchstoneQuery(ctx, name, schema, "array_contains(tag_ids, 8791004)", enabled, action)
					r.Equal(expected["array"], got)
					feature := "filter_exec_path=brute_force"
					if indexed {
						feature = "scalar_index_type=INVERTED"
					}
					r.Greater(s.executionFeatures(ctx)[feature], before[feature], "native path indexed="+strconv.FormatBool(indexed))
				}
			}
			status, err = c.ReleaseCollection(ctx, &milvuspb.ReleaseCollectionRequest{CollectionName: name})
			r.NoError(merr.CheckRPCCall(status, err))
		}
		status, err = c.DropCollection(ctx, &milvuspb.DropCollectionRequest{CollectionName: name})
		r.NoError(merr.CheckRPCCall(status, err))
	}
}

func (s *EmptyPipelineSuite) touchstoneQuery(ctx context.Context, name string, schema *schemapb.CollectionSchema, query string, enabled bool, action string) []int64 {
	dirs := s.arm(schema, action, query)
	result, err := s.Cluster.MilvusClient.Query(ctx, &milvuspb.QueryRequest{CollectionName: name, Expr: query, OutputFields: []string{"pk"}, ConsistencyLevel: commonpb.ConsistencyLevel_Strong})
	s.Require().NoError(merr.CheckRPCCall(result, err))
	s.translated(dirs, enabled, orchestration.OutcomeTranslated)
	if enabled {
		for _, dir := range dirs {
			s.Require().True(s.receipt(dir, "completed").SourceUnchanged)
			if action != "observe" {
				data, err := os.ReadFile(filepath.Join(dir, "rewritten.json"))
				s.Require().NoError(err)
				var receipt struct {
					Passes      int
					Before      int   `json:"before_nodes"`
					After       int   `json:"after_nodes"`
					Nanoseconds int64 `json:"rewrite_ns"`
				}
				s.Require().NoError(json.Unmarshal(data, &receipt))
				count, err := strconv.Atoi(action[len("optimize:"):])
				s.Require().NoError(err)
				s.Require().Equal(count, receipt.Passes)
				s.T().Logf("rewrite passes=%d nodes=%d->%d duration=%dns", receipt.Passes, receipt.Before, receipt.After, receipt.Nanoseconds)
			}
		}
	}
	ids := append([]int64{}, result.GetFieldsData()[0].GetScalars().GetLongData().GetData()...)
	sort.Slice(ids, func(i, j int) bool { return ids[i] < ids[j] })
	return ids
}

// Uses compact nullable payloads while retaining empty arrays as valid rows.
func touchstoneFields(rows []touchstone.Row, nullableArticle bool) []*schemapb.FieldData {
	var pks, reviews, brands []int64
	var articles []string
	var vectors []float32
	articleValid, reviewValid, brandValid := make([]bool, len(rows)), make([]bool, len(rows)), make([]bool, len(rows))
	arrayData := []*schemapb.ArrayArray{{ElementType: schemapb.DataType_Int64}, {ElementType: schemapb.DataType_Int64}, {ElementType: schemapb.DataType_Int64}}
	arrayValid := [][]bool{make([]bool, len(rows)), make([]bool, len(rows)), make([]bool, len(rows))}
	for i, row := range rows {
		pks = append(pks, row.PK)
		vectors = append(vectors, 1, 0)
		if row.Article != nil {
			articles = append(articles, *row.Article)
			articleValid[i] = true
		}
		if row.Review != nil {
			reviews = append(reviews, *row.Review)
			reviewValid[i] = true
		}
		if row.Brand != nil {
			brands = append(brands, *row.Brand)
			brandValid[i] = true
		}
		for j, values := range [][]int64{row.Tags, row.Categories, row.Malls} {
			if values != nil {
				arrayValid[j][i] = true
				arrayData[j].Data = append(arrayData[j].Data, &schemapb.ScalarField{Data: &schemapb.ScalarField_LongData{LongData: &schemapb.LongArray{Data: values}}})
			}
		}
	}
	if !nullableArticle {
		articleValid = nil
	}
	fields := []*schemapb.FieldData{
		{FieldName: "pk", Type: schemapb.DataType_Int64, Field: &schemapb.FieldData_Scalars{Scalars: &schemapb.ScalarField{Data: &schemapb.ScalarField_LongData{LongData: &schemapb.LongArray{Data: pks}}}}},
		{FieldName: "article_type", Type: schemapb.DataType_VarChar, ValidData: articleValid, Field: &schemapb.FieldData_Scalars{Scalars: &schemapb.ScalarField{Data: &schemapb.ScalarField_StringData{StringData: &schemapb.StringArray{Data: articles}}}}},
		{FieldName: "is_review", Type: schemapb.DataType_Int64, ValidData: reviewValid, Field: &schemapb.FieldData_Scalars{Scalars: &schemapb.ScalarField{Data: &schemapb.ScalarField_LongData{LongData: &schemapb.LongArray{Data: reviews}}}}},
		{FieldName: "brand_id", Type: schemapb.DataType_Int64, ValidData: brandValid, Field: &schemapb.FieldData_Scalars{Scalars: &schemapb.ScalarField{Data: &schemapb.ScalarField_LongData{LongData: &schemapb.LongArray{Data: brands}}}}},
		{FieldName: "vector", Type: schemapb.DataType_FloatVector, Field: &schemapb.FieldData_Vectors{Vectors: &schemapb.VectorField{Dim: 2, Data: &schemapb.VectorField_FloatVector{FloatVector: &schemapb.FloatArray{Data: vectors}}}}},
	}
	for i, name := range []string{"tag_ids", "category_ids", "mall_ids"} {
		fields = append(fields, &schemapb.FieldData{FieldName: name, Type: schemapb.DataType_Array, ValidData: arrayValid[i], Field: &schemapb.FieldData_Scalars{Scalars: &schemapb.ScalarField{Data: &schemapb.ScalarField_ArrayData{ArrayData: arrayData[i]}}}})
	}
	return fields
}

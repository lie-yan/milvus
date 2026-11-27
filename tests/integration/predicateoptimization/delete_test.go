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
	"encoding/json"
	"os"
	"path/filepath"
	"sort"
	"strconv"

	"google.golang.org/protobuf/proto"

	"github.com/milvus-io/milvus-proto/go-api/v3/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v3/milvuspb"
	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
	"github.com/milvus-io/milvus/client/v3/membership/sbbf"
	"github.com/milvus-io/milvus/internal/parser/planparserv2/predicateadapter/orchestration"
	"github.com/milvus-io/milvus/internal/parser/planparserv2/predicateadapter/tests/touchstone"
	"github.com/milvus-io/milvus/pkg/v3/util/funcutil"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
	"github.com/milvus-io/milvus/pkg/v3/util/metric"
	"github.com/milvus-io/milvus/tests/integration"
)

// Compares actual deletion effects against independent row expectations on fresh collections.
func (s *EmptyPipelineSuite) TestOptimizedDeleteParity() {
	ctx, c, r := s.Cluster.GetContext(), s.Cluster.MilvusClient, s.Require()
	cases, err := touchstone.Cases(false)
	r.NoError(err)
	var rows []touchstone.Row
	all := []int64{}
	for _, tc := range cases {
		rows = append(rows, tc.Row)
		all = append(all, tc.Row.PK)
	}
	sort.Slice(all, func(i, j int) bool { return all[i] < all[j] })
	builder, err := sbbf.NewBuilder(3, 0.001)
	r.NoError(err)
	builder.AddInt64(1)
	for _, tc := range []struct {
		name, expr string
		matches    func(touchstone.Case) bool
		outcome    orchestration.Outcome
	}{
		{"primary_keys", "pk in [1, 2]", func(c touchstone.Case) bool { return c.Row.PK == 1 || c.Row.PK == 2 }, orchestration.OutcomeTranslated},
		{"nullable", "is_review == 8795001", func(c touchstone.Case) bool { return c.Row.Review != nil && *c.Row.Review == 8795001 }, orchestration.OutcomeTranslated},
		{"touchstone", touchstone.Query, func(c touchstone.Case) bool { return c.Expected == "TRUE" }, orchestration.OutcomeTranslated},
		{"unsupported", "pk + 1 < 5", func(c touchstone.Case) bool { return c.Row.PK+1 < 5 }, orchestration.OutcomeUnsupported},
	} {
		for _, enabled := range []bool{false, true} {
			s.Run(tc.name+"/enabled="+strconv.FormatBool(enabled), func() {
				r := s.Require()
				name := "predicate_delete_" + funcutil.GenRandomStr()
				schema := touchstone.Schema(name, false)
				encoded, err := proto.Marshal(schema)
				r.NoError(err)
				status, err := c.CreateCollection(ctx, &milvuspb.CreateCollectionRequest{CollectionName: name, Schema: encoded, ShardsNum: 1})
				r.NoError(merr.CheckRPCCall(status, err))
				defer func() {
					status, err := c.DropCollection(ctx, &milvuspb.DropCollectionRequest{CollectionName: name})
					r.NoError(merr.CheckRPCCall(status, err))
				}()
				inserted, err := c.Insert(ctx, &milvuspb.InsertRequest{CollectionName: name, FieldsData: touchstone.Fields(rows, false), NumRows: uint32(len(rows)), HashKeys: integration.GenerateHashKeys(len(rows))})
				r.NoError(merr.CheckRPCCall(inserted, err))
				flushed, err := c.Flush(ctx, &milvuspb.FlushRequest{CollectionNames: []string{name}})
				r.NoError(merr.CheckRPCCall(flushed, err))
				s.WaitForFlush(ctx, flushed.GetFlushCollSegIDs()[name].GetData(), flushed.GetCollFlushTs()[name], "", name)
				status, err = c.CreateIndex(ctx, &milvuspb.CreateIndexRequest{CollectionName: name, FieldName: "vector", ExtraParams: integration.ConstructIndexParam(2, integration.IndexFaissIDMap, metric.L2)})
				r.NoError(merr.CheckRPCCall(status, err))
				s.WaitForIndexBuilt(ctx, name, "vector")
				status, err = c.LoadCollection(ctx, &milvuspb.LoadCollectionRequest{CollectionName: name})
				r.NoError(merr.CheckRPCCall(status, err))
				s.WaitForLoad(ctx, name)
				described, err := c.DescribeCollection(ctx, &milvuspb.DescribeCollectionRequest{CollectionName: name})
				r.NoError(merr.CheckRPCCall(described, err))
				schema = described.GetSchema()
				s.setEnabled(ctx, enabled)
				readIDs := func() []int64 {
					result, err := c.Query(ctx, &milvuspb.QueryRequest{CollectionName: name, Expr: "", OutputFields: []string{"pk"}, ConsistencyLevel: commonpb.ConsistencyLevel_Strong, QueryParams: []*commonpb.KeyValuePair{{Key: "limit", Value: "100"}}})
					r.NoError(merr.CheckRPCCall(result, err))
					ids := append([]int64{}, result.GetFieldsData()[0].GetScalars().GetLongData().GetData()...)
					sort.Slice(ids, func(i, j int) bool { return ids[i] < ids[j] })
					return ids
				}
				r.Equal(all, readIDs())
				// Rejections must leave the rows intact, before the successful mutation.
				if tc.name == "primary_keys" {
					for _, expr := range []string{" ", "true", "membership_match(pk, {bf}, type=bloom)"} {
						result, err := c.Delete(ctx, &milvuspb.DeleteRequest{CollectionName: name, Expr: expr, ExprTemplateValues: map[string]*schemapb.TemplateValue{"bf": {Val: &schemapb.TemplateValue_BytesVal{BytesVal: builder.Marshal()}}}})
						r.NoError(err)
						r.False(merr.Ok(result.GetStatus()))
						if expr == "membership_match(pk, {bf}, type=bloom)" {
							r.Contains(result.GetStatus().GetReason(), "approximate bloom filter blob")
						} else {
							r.Contains(result.GetStatus().GetReason(), "empty or always true")
						}
						r.Equal(all, readIDs())
					}
					if enabled {
						expr := "not (pk == 1 and pk == 2)"
						dirs := s.arm(schema, "optimize:5", expr)
						result, err := c.Delete(ctx, &milvuspb.DeleteRequest{CollectionName: name, Expr: expr})
						r.NoError(err)
						r.False(merr.Ok(result.GetStatus()))
						r.Contains(result.GetStatus().GetReason(), "empty or always true")
						s.translated(dirs, true, orchestration.OutcomeTranslated)
						r.Equal(all, readIDs())
					}
				}
				expectedDeleted, expectedSurvivors := []int64{}, []int64{}
				for _, row := range cases {
					if tc.matches(row) {
						expectedDeleted = append(expectedDeleted, row.Row.PK)
					} else {
						expectedSurvivors = append(expectedSurvivors, row.Row.PK)
					}
				}
				sort.Slice(expectedDeleted, func(i, j int) bool { return expectedDeleted[i] < expectedDeleted[j] })
				sort.Slice(expectedSurvivors, func(i, j int) bool { return expectedSurvivors[i] < expectedSurvivors[j] })
				dirs := s.arm(schema, "optimize:5", tc.expr)
				deleted, err := c.Delete(ctx, &milvuspb.DeleteRequest{CollectionName: name, Expr: tc.expr})
				r.NoError(merr.CheckRPCCall(deleted, err))
				r.Equal(int64(len(expectedDeleted)), deleted.GetDeleteCnt())
				s.translated(dirs, enabled, tc.outcome)
				if enabled {
					for _, dir := range dirs {
						r.True(s.receipt(dir, "completed").SourceUnchanged)
					}
					if tc.name == "touchstone" {
						data, err := os.ReadFile(filepath.Join(dirs[0], "rewritten.json"))
						r.NoError(err)
						var receipt struct {
							Passes int
							Before int `json:"before_nodes"`
							After  int `json:"after_nodes"`
						}
						r.NoError(json.Unmarshal(data, &receipt))
						r.Equal(5, receipt.Passes)
						r.Less(receipt.After, receipt.Before, "must actually simplify, not just round-trip")
					}
				}
				// Strong consistency includes the completed delete before observing survivors.
				survivors := readIDs()
				r.Equal(expectedSurvivors, survivors)
				remaining := map[int64]bool{}
				for _, id := range survivors {
					remaining[id] = true
				}
				actualDeleted := []int64{}
				for _, id := range all {
					if !remaining[id] {
						actualDeleted = append(actualDeleted, id)
					}
				}
				r.Equal(expectedDeleted, actualDeleted)
				s.T().Logf("delete enabled=%t rows=%d deleted=%v survivors=%v", enabled, len(rows), actualDeleted, survivors)
			})
		}
	}
	s.setEnabled(ctx, false)
}

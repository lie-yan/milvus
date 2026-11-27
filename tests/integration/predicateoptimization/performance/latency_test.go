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

package performance

import (
	"context"
	"encoding/json"
	"net/http"
	"sort"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"
	"google.golang.org/protobuf/proto"

	"github.com/milvus-io/milvus-proto/go-api/v3/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v3/milvuspb"
	"github.com/milvus-io/milvus/internal/parser/planparserv2/predicateadapter/tests/touchstone"
	"github.com/milvus-io/milvus/pkg/v3/util/funcutil"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
	"github.com/milvus-io/milvus/pkg/v3/util/metric"
	"github.com/milvus-io/milvus/tests/integration"
	"github.com/milvus-io/milvus/tests/integration/cluster"
)

// Measures the ordinary server binary, without the parent suite's observation overlay.
type latencySuite struct{ integration.MiniClusterSuite }

func (s *latencySuite) SetupSuite() {
	s.WithClusterOptions(cluster.WithMetricsPorts())
	s.WithMilvusConfig("indexCoord.segment.minSegmentNumRowsToEnableIndex", "64")
	s.MiniClusterSuite.SetupSuite()
}

func (s *latencySuite) setEnabled(ctx context.Context, enabled bool) {
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
		var config map[string]string
		if resp.StatusCode != http.StatusOK || json.NewDecoder(resp.Body).Decode(&config) != nil {
			return false
		}
		for key, got := range config {
			if strings.EqualFold(strings.ReplaceAll(key, ".", ""), "proxyenablePredicateOptimization") && strings.HasPrefix(got, value+"[") {
				return true
			}
		}
		return false
	}, 15*time.Second, 100*time.Millisecond)
}

func (s *latencySuite) TestQueryLatency() {
	ctx, c, r := s.Cluster.GetContext(), s.Cluster.MilvusClient, s.Require()
	name := "predicate_latency_" + funcutil.GenRandomStr()
	encoded, err := proto.Marshal(touchstone.Schema(name, false))
	r.NoError(err)
	status, err := c.CreateCollection(ctx, &milvuspb.CreateCollectionRequest{CollectionName: name, Schema: encoded, ShardsNum: 1})
	r.NoError(merr.CheckRPCCall(status, err))
	cases, err := touchstone.Cases(false)
	r.NoError(err)
	var rows []touchstone.Row
	for copy := int64(0); copy < 256; copy++ {
		for _, tc := range cases {
			row := tc.Row
			row.PK += copy * 1000
			rows = append(rows, row)
		}
	}
	inserted, err := c.Insert(ctx, &milvuspb.InsertRequest{CollectionName: name, FieldsData: touchstone.Fields(rows, false), NumRows: uint32(len(rows)), HashKeys: integration.GenerateHashKeys(len(rows))})
	r.NoError(merr.CheckRPCCall(inserted, err))
	flushed, err := c.Flush(ctx, &milvuspb.FlushRequest{CollectionNames: []string{name}})
	r.NoError(merr.CheckRPCCall(flushed, err))
	s.WaitForFlush(ctx, flushed.GetFlushCollSegIDs()[name].GetData(), flushed.GetCollFlushTs()[name], "", name)
	status, err = c.CreateIndex(ctx, &milvuspb.CreateIndexRequest{CollectionName: name, FieldName: "vector", ExtraParams: integration.ConstructIndexParam(2, integration.IndexFaissIDMap, metric.L2)})
	r.NoError(merr.CheckRPCCall(status, err))
	s.WaitForIndexBuilt(ctx, name, "vector")
	for _, indexed := range []bool{false, true} {
		if indexed {
			for _, field := range []string{"pk", "article_type", "is_review", "brand_id", "tag_ids", "category_ids", "mall_ids"} {
				status, err = c.CreateIndex(ctx, &milvuspb.CreateIndexRequest{CollectionName: name, FieldName: field, ExtraParams: []*commonpb.KeyValuePair{{Key: "index_type", Value: "INVERTED"}}})
				r.NoError(merr.CheckRPCCall(status, err))
				s.WaitForIndexBuilt(ctx, name, field)
			}
		}
		status, err = c.LoadCollection(ctx, &milvuspb.LoadCollectionRequest{CollectionName: name})
		r.NoError(merr.CheckRPCCall(status, err))
		s.WaitForLoad(ctx, name)
		for _, workload := range touchstone.PerformanceWorkloads() {
			request := &milvuspb.QueryRequest{CollectionName: name, Expr: workload.Query, OutputFields: []string{"count(*)"}, ConsistencyLevel: commonpb.ConsistencyLevel_Strong}
			samples := map[bool][]int64{}
			var expected int64
			query := func() (int64, int64) {
				start := time.Now()
				result, err := c.Query(ctx, request)
				elapsed := time.Since(start).Nanoseconds()
				r.NoError(merr.CheckRPCCall(result, err))
				return result.GetFieldsData()[0].GetScalars().GetLongData().GetData()[0], elapsed
			}
			// ABBA blocks balance order; config propagation and five warmups are untimed.
			for block, enabled := range []bool{false, true, true, false} {
				s.setEnabled(ctx, enabled)
				for warm := 0; warm < 5; warm++ {
					count, _ := query()
					if block == 0 && warm == 0 {
						expected = count
					}
					r.Equal(expected, count)
				}
				var blockSamples []int64
				for i := 0; i < 25; i++ {
					count, ns := query()
					r.Equal(expected, count)
					samples[enabled] = append(samples[enabled], ns)
					blockSamples = append(blockSamples, ns)
				}
				raw, err := json.Marshal(blockSamples)
				r.NoError(err)
				s.T().Logf("LATENCY_SAMPLES workload=%s indexed=%t enabled=%t block=%d rows=%d count=%d ns=%s", workload.Name, indexed, enabled, block, len(rows), expected, raw)
			}
			for _, enabled := range []bool{false, true} {
				ns := samples[enabled]
				sort.Slice(ns, func(i, j int) bool { return ns[i] < ns[j] })
				s.T().Logf("LATENCY_SUMMARY workload=%s indexed=%t enabled=%t n=%d p50_ns=%d p95_ns=%d", workload.Name, indexed, enabled, len(ns), ns[len(ns)/2], ns[(95*len(ns)+99)/100-1])
			}
		}
		status, err = c.ReleaseCollection(ctx, &milvuspb.ReleaseCollectionRequest{CollectionName: name})
		r.NoError(merr.CheckRPCCall(status, err))
	}
	status, err = c.DropCollection(ctx, &milvuspb.DropCollectionRequest{CollectionName: name})
	r.NoError(merr.CheckRPCCall(status, err))
}

func TestPredicateLatency(t *testing.T) { suite.Run(t, new(latencySuite)) }

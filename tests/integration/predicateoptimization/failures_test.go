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
	"os"
	"path/filepath"
	"time"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/proto"

	"github.com/milvus-io/milvus-proto/go-api/v3/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v3/milvuspb"
	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
	"github.com/milvus-io/milvus/internal/parser/planparserv2/predicateadapter/orchestration"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
)

// Exercise real server abort projection and client cancellation during a paused translation.
func (s *EmptyPipelineSuite) verifyFailures(ctx context.Context, name string, schema *schemapb.CollectionSchema, search *milvuspb.SearchRequest) {
	r := s.Require()
	s.setEnabled(ctx, true)
	c := s.Cluster.MilvusClient
	calls := []struct {
		name, source string
		call         func() (*commonpb.Status, error)
	}{
		{"query", "i > 7", func() (*commonpb.Status, error) {
			v, e := c.Query(ctx, &milvuspb.QueryRequest{CollectionName: name, Expr: "i > 7", OutputFields: []string{"pk"}})
			return v.GetStatus(), e
		}},
		{"search", "i > 7", func() (*commonpb.Status, error) {
			v, e := c.Search(ctx, proto.Clone(search).(*milvuspb.SearchRequest))
			return v.GetStatus(), e
		}},
		{"scorer", "i < 12", func() (*commonpb.Status, error) {
			request := proto.Clone(search).(*milvuspb.SearchRequest)
			request.Dsl = "i + 1 < 20"
			request.FunctionScore = &schemapb.FunctionScore{Functions: []*schemapb.FunctionSchema{{Name: "boost", Type: schemapb.FunctionType_Rerank, Params: []*commonpb.KeyValuePair{{Key: "reranker", Value: "boost"}, {Key: "filter", Value: "i < 12"}, {Key: "weight", Value: "2"}}}}}
			v, e := c.Search(ctx, request)
			return v.GetStatus(), e
		}},
	}
	for _, tc := range calls {
		dirs := s.arm(schema, "unsafe", tc.source)
		wire, err := tc.call()
		r.NoError(err, tc.name)
		done := s.receipt(dirs[0], "completed")
		r.True(done.Abort, tc.name)
		r.Equal(merr.Code(merr.ErrServiceInternal), done.Code)
		r.Equal(done.Code, wire.GetCode(), tc.name)
		r.NotEqual("true", wire.GetExtraInfo()["is_input_error"], tc.name)
		r.Equal(merr.Status(merr.ErrServiceInternal).GetRetriable(), wire.GetRetriable(), tc.name)
	}
	dirs := s.arm(schema, "pause", "i > 7")
	defer os.WriteFile(filepath.Join(dirs[0], "release"), nil, 0600)
	requestCtx, cancel := context.WithCancel(ctx)
	defer cancel()
	finished := make(chan error, 1)
	go func() {
		_, err := c.Query(requestCtx, &milvuspb.QueryRequest{CollectionName: name, Expr: "i > 7", OutputFields: []string{"pk"}})
		finished <- err
	}()
	entered := s.receipt(dirs[0], "entered")
	r.True(entered.IndependentContext)
	cancel()
	select {
	case err := <-finished:
		r.Equal(codes.Canceled, status.Code(err))
	case <-time.After(10 * time.Second):
		r.FailNow("client did not observe cancellation")
	}
	_, err := os.Stat(filepath.Join(dirs[0], "completed.json"))
	r.True(os.IsNotExist(err), "translation must still be paused when the client returns")
	r.NoError(os.WriteFile(filepath.Join(dirs[0], "release"), nil, 0600))
	done := s.receipt(dirs[0], "completed")
	r.False(done.Abort)
	r.Equal(orchestration.OutcomeTranslated, done.Outcome)
	// The canceled attempt must not poison the next request.
	dirs = s.arm(schema, "observe", "i > 7")
	result, err := c.Query(ctx, &milvuspb.QueryRequest{CollectionName: name, Expr: "i > 7", OutputFields: []string{"count(*)"}})
	r.NoError(merr.CheckRPCCall(result, err))
	r.Equal([]int64{2040}, result.GetFieldsData()[0].GetScalars().GetLongData().GetData())
	s.translated(dirs, true, orchestration.OutcomeTranslated)
}

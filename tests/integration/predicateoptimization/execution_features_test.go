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

	"github.com/milvus-io/milvus/internal/featureusage"
	"github.com/milvus-io/milvus/pkg/v3/proto/internalpb"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
	"github.com/milvus-io/milvus/pkg/v3/util/typeutil"
)

// Execution bits originate in native operators and are merged into the proxy report.
func (s *EmptyPipelineSuite) executionFeatures(ctx context.Context) map[string]int64 {
	report, err := s.Cluster.MixCoordClient.GetFeatureUsage(ctx, &internalpb.GetFeatureUsageRequest{})
	s.Require().NoError(merr.CheckRPCCall(report, err))
	counters := map[string]int64{}
	for _, node := range report.GetNodes() {
		s.Require().True(node.GetReachable())
		if node.GetRole() != typeutil.ProxyRole {
			continue
		}
		for _, entry := range node.GetEntries() {
			if entry.GetGroup() == featureusage.GroupRequest {
				counters[entry.GetName()] += entry.GetValue()
			}
		}
	}
	return counters
}

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
	"crypto/sha256"
	_ "embed"
	"encoding/json"
	"fmt"
	"go/format"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"google.golang.org/protobuf/proto"

	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
	"github.com/milvus-io/milvus/internal/parser/planparserv2"
	"github.com/milvus-io/milvus/internal/parser/planparserv2/predicateadapter/orchestration"
	"github.com/milvus-io/milvus/pkg/v3/util/typeutil"
)

//go:embed probe.go.txt
var probeSource string

// Build an observed server without modifying production sources or the normal binary.
func TestMain(m *testing.M) { os.Exit(runObserved(m)) }

func runObserved(m *testing.M) int {
	root := os.Getenv("MILVUS_WORK_DIR")
	if root == "" {
		fmt.Fprintln(os.Stderr, "MILVUS_WORK_DIR is required")
		return 1
	}
	output, err := os.MkdirTemp("", "predicate-observed-")
	if err != nil {
		panic(err)
	}
	defer os.RemoveAll(output)
	original := filepath.Join(root, "internal/parser/planparserv2/predicateadapter/integration/integration.go")
	source, err := os.ReadFile(original)
	if err != nil {
		panic(err)
	}
	if strings.Count(string(source), "orchestration.RoundTrip(") != 1 {
		panic("adapter observation seam changed")
	}
	rewritten := strings.Replace(string(source), "orchestration.RoundTrip(", "probeRoundTrip(", 1)
	rewritten = strings.Replace(rewritten, "import (", `import (
 "crypto/sha256"
 "encoding/json"
 "fmt"
 "os"
 "path/filepath"
 "time"
 "google.golang.org/protobuf/proto"
`, 1) + probeSource
	formatted, err := format.Source([]byte(rewritten))
	if err != nil {
		panic(err)
	}
	replacement := filepath.Join(output, "integration.go")
	if err = os.WriteFile(replacement, formatted, 0600); err != nil {
		panic(err)
	}
	overlay, err := json.Marshal(map[string]any{"Replace": map[string]string{original: replacement}})
	if err != nil {
		panic(err)
	}
	overlayPath := filepath.Join(output, "overlay.json")
	if err = os.WriteFile(overlayPath, overlay, 0600); err != nil {
		panic(err)
	}
	if err = os.MkdirAll(filepath.Join(output, "bin"), 0700); err != nil {
		panic(err)
	}
	if err = os.Symlink(filepath.Join(root, "configs"), filepath.Join(output, "configs")); err != nil {
		panic(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Minute)
	defer cancel()
	cmd := exec.CommandContext(ctx, "make", "build-go", "INSTALL_PATH="+filepath.Join(output, "bin"))
	cmd.Dir = root
	cmd.Env = append(cmd.Environ(), "GOFLAGS="+os.Getenv("GOFLAGS")+" -overlay="+overlayPath)
	cmd.Stdout, cmd.Stderr = os.Stdout, os.Stderr
	if err = cmd.Run(); err != nil {
		fmt.Fprintln(os.Stderr, "build observed fixture:", err)
		return 1
	}
	if err = os.Setenv("MILVUS_WORK_DIR", output); err != nil {
		panic(err)
	}
	return m.Run()
}

type attemptControl struct {
	Directory string
	Action    string
}
type attemptReceipt struct {
	SourceUnchanged    bool `json:"source_unchanged"`
	PID                int
	IndependentContext bool `json:"independent_context"`
	Outcome            orchestration.Outcome
	Abort              bool
	Code               int32
}

// Select exact post-legacy expressions; one-shot receipts exclude unrelated requests.
func (s *EmptyPipelineSuite) arm(schema *schemapb.CollectionSchema, action string, sources ...string) []string {
	helper, err := typeutil.CreateSchemaHelper(schema)
	s.Require().NoError(err)
	controls := map[string]attemptControl{}
	var dirs []string
	for _, source := range sources {
		if strings.TrimSpace(source) == "" {
			continue
		}
		expr, err := planparserv2.ParseExpr(helper, source, nil)
		s.Require().NoError(err)
		encoded, err := proto.MarshalOptions{Deterministic: true}.Marshal(expr)
		s.Require().NoError(err)
		dir, err := os.MkdirTemp(s.controlDir, "attempt-")
		s.Require().NoError(err)
		controls[fmt.Sprintf("%x", sha256.Sum256(encoded))] = attemptControl{dir, action}
		dirs = append(dirs, dir)
	}
	data, err := json.Marshal(controls)
	s.Require().NoError(err)
	path := filepath.Join(s.controlDir, "control.json")
	s.Require().NoError(os.WriteFile(path+".tmp", data, 0600))
	s.Require().NoError(os.Rename(path+".tmp", path))
	return dirs
}

func (s *EmptyPipelineSuite) receipt(dir, name string) attemptReceipt {
	var receipt attemptReceipt
	s.Require().Eventually(func() bool {
		data, err := os.ReadFile(filepath.Join(dir, name+".json"))
		return err == nil && json.Unmarshal(data, &receipt) == nil
	}, 10*time.Second, 20*time.Millisecond, "missing %s receipt for %s", name, dir)
	s.Require().Equal(s.Cluster.DefaultProxy().PID(), receipt.PID)
	return receipt
}

func (s *EmptyPipelineSuite) translated(dirs []string, enabled bool, outcomes ...orchestration.Outcome) {
	for i, dir := range dirs {
		if !enabled {
			_, err := os.Stat(filepath.Join(dir, "claim"))
			s.Require().True(os.IsNotExist(err))
			continue
		}
		entered := s.receipt(dir, "entered")
		s.Require().True(entered.IndependentContext)
		done := s.receipt(dir, "completed")
		s.Require().False(done.Abort)
		s.Require().Equal(outcomes[i], done.Outcome)
	}
}

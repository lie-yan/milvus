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

// Package touchstone supplies the fixed query and independent row expectations for empty-pipeline tests.
package touchstone

import (
	_ "embed"
	"encoding/json"

	"github.com/milvus-io/milvus-proto/go-api/v3/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
)

// Exact motivating query, with no optimizer passes or reduced-expression substitution.
//
//go:embed query.txt
var Query string

//go:embed dataset.json
var dataset []byte

// One recorded input and its independent three-valued expectation.
type Case struct {
	Name     string `json:"case"`
	Schema   string `json:"schema"`
	Row      Row    `json:"row"`
	Expected string `json:"expected_input"`
}

// Nil pointers/slices denote NULL; nonnil empty slices denote present empty arrays.
type Row struct {
	PK         int64   `json:"pk"`
	Article    *string `json:"article_type"`
	Review     *int64  `json:"is_review"`
	Brand      *int64  `json:"brand_id"`
	Tags       []int64 `json:"tag_ids"`
	Categories []int64 `json:"category_ids"`
	Malls      []int64 `json:"mall_ids"`
}

// Selects valid rows for the schema, including the nullable-article counterexamples when allowed.
func Cases(nullableArticle bool) ([]Case, error) {
	var all []Case
	if err := json.Unmarshal(dataset, &all); err != nil {
		return nil, err
	}
	selected := make([]Case, 0, len(all))
	for _, c := range all {
		if nullableArticle || c.Row.Article != nil {
			selected = append(selected, c)
		}
	}
	return selected, nil
}

// Defines the scalar fixture and a two-dimensional vector for native collection loading.
func Schema(name string, nullableArticle bool) *schemapb.CollectionSchema {
	fields := []*schemapb.FieldSchema{
		{FieldID: 100, Name: "pk", DataType: schemapb.DataType_Int64, IsPrimaryKey: true},
		{FieldID: 101, Name: "article_type", DataType: schemapb.DataType_VarChar, Nullable: nullableArticle, TypeParams: []*commonpb.KeyValuePair{{Key: "max_length", Value: "32"}}},
		{FieldID: 102, Name: "is_review", DataType: schemapb.DataType_Int64, Nullable: true},
		{FieldID: 103, Name: "brand_id", DataType: schemapb.DataType_Int64, Nullable: true},
	}
	for i, field := range []string{"tag_ids", "category_ids", "mall_ids"} {
		fields = append(fields, &schemapb.FieldSchema{FieldID: int64(104 + i), Name: field, DataType: schemapb.DataType_Array, ElementType: schemapb.DataType_Int64, Nullable: true, TypeParams: []*commonpb.KeyValuePair{{Key: "max_capacity", Value: "32"}}})
	}
	fields = append(fields, &schemapb.FieldSchema{FieldID: 107, Name: "vector", DataType: schemapb.DataType_FloatVector, TypeParams: []*commonpb.KeyValuePair{{Key: "dim", Value: "2"}}})
	return &schemapb.CollectionSchema{Name: name, Fields: fields}
}

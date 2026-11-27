package touchstone

import "github.com/milvus-io/milvus-proto/go-api/v3/schemapb"

// Uses compact nullable payloads while retaining empty arrays as valid rows.
func Fields(rows []Row, nullableArticle bool) []*schemapb.FieldData {
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

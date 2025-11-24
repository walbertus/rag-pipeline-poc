package tools

import (
	"context"
	"errors"
	"log/slog"

	"github.com/gopaytech/rag-pipeline-poc/retriever/internal/model_garden"
	"github.com/milvus-io/milvus/client/v2/entity"
	"github.com/milvus-io/milvus/client/v2/index"
	"github.com/milvus-io/milvus/client/v2/milvusclient"
	"github.com/modelcontextprotocol/go-sdk/mcp"
)

type Tools struct {
	_           struct{}
	database    *milvusclient.Client
	logger      *slog.Logger
	modelGarden *model_garden.ModelGarden
}

func RegisterTools(logger *slog.Logger, server *mcp.Server, database *milvusclient.Client, modelGarden *model_garden.ModelGarden) *Tools {
	t := &Tools{
		database:    database,
		logger:      logger.WithGroup("tools"),
		modelGarden: modelGarden,
	}

	mcp.AddTool(server, QueryTool, t.Query)

	return t
}

func (t *Tools) Query(ctx context.Context,
	req *mcp.CallToolRequest,
	input QueryInput,
) (*mcp.CallToolResult, *QueryOutput, error) {
	t.logger.LogAttrs(ctx,
		slog.LevelInfo,
		"received query tool request",
		slog.String("query", input.Query),
		slog.Any("req", req),
	)

	if ok, err := t.database.HasCollection(ctx, milvusclient.NewHasCollectionOption("pdf_collection")); err != nil {
		return nil, nil, err
	} else if !ok {
		return nil, nil, errors.New("collection does not exist")
	}

	denseVector := make([]entity.Vector, 0, 1)
	if resp, err := t.modelGarden.Vectorize(ctx, []string{input.Query}); err != nil {
		return nil, nil, err
	} else {
		denseVector = append(denseVector, entity.FloatVector(resp[0]))
	}
	param := index.NewSparseAnnParam()
	param.WithDropRatio(0.2)
	rs, err := t.database.HybridSearch(ctx,
		milvusclient.NewHybridSearchOption(
			"pdf_collection",
			2,
			milvusclient.NewAnnRequest(
				"vector_dense",
				2,
				denseVector...,
			).WithANNSField("vector_dense"),
			milvusclient.NewAnnRequest(
				"vector_sparse",
				2,
				entity.Text(input.Query),
			).WithANNSField("vector_sparse").WithAnnParam(param),
		).
			WithOutputFields("text", "metadata").
			WithReranker(milvusclient.NewRRFReranker()),
	)
	if err != nil {
		return nil, nil, err
	}

	for _, r := range rs {
		t.logger.LogAttrs(ctx,
			slog.LevelInfo,
			"field info",
			slog.Any("C", r.GetColumn("text").FieldData().GetScalars().GetStringData().GetData()),
			slog.Any("D", r.GetColumn("metadata").FieldData().GetScalars().GetJsonData().GetData()),
		)
	}

	return nil, &QueryOutput{
		TopK: []string{"result1", "result2", "result3"},
	}, nil
}

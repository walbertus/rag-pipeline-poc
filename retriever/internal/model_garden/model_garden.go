package model_garden

import (
	"bytes"
	"context"
	"encoding/json"
	"log/slog"
	"net/http"
	"time"
)

type (
	ModelGarden struct {
		_              struct{}
		client         *http.Client
		logger         *slog.Logger
		modelGardenURL string
		modelName      string
	}
	EmbeddingsRequest struct {
		_              struct{}
		Model          string   `json:"model"`
		Input          []string `json:"input"`
		EncodingFormat string   `json:"encoding_format"`
	}
	EmbeddingsResponse struct {
		_     struct{}
		Model string `json:"model"`
		Data  []struct {
			Embedding []float64 `json:"embedding"`
			Index     int       `json:"index"`
			Object    string    `json:"object"`
		} `json:"data"`
		Object string `json:"object"`
		Usage  struct {
			PromptTokens           int            `json:"prompt_tokens"`
			TotalTokens            int            `json:"total_tokens"`
			CompletionTokens       int            `json:"completion_tokens"`
			PrompTokenDetails      map[string]any `json:"promp_token_details"`
			CompletionTokenDetails map[string]any `json:"completion_token_details"`
		} `json:"usage"`
	}
)

func NewModelGarden(
	client *http.Client,
	logger *slog.Logger,
	modelGardenURL string,
	modelName string,
) *ModelGarden {
	return &ModelGarden{
		client:         client,
		logger:         logger.WithGroup("model_garden"),
		modelGardenURL: modelGardenURL,
		modelName:      modelName,
	}
}

func (m *ModelGarden) Vectorize(ctx context.Context, text []string) ([][]float32, error) {
	ctx, cancel := context.WithTimeout(ctx, time.Minute)
	defer cancel()

	b, err := json.Marshal(&EmbeddingsRequest{
		Model:          m.modelName,
		Input:          text,
		EncodingFormat: "float",
	})
	if err != nil {
		return nil, err
	}

	req, err := http.NewRequestWithContext(ctx,
		http.MethodPost,
		m.modelGardenURL,
		bytes.NewReader(b),
	)
	if err != nil {
		return nil, err
	}
	defer req.Body.Close()
	req.Header.Set("Content-Type", "application/json")

	resp, err := m.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var res EmbeddingsResponse
	if err := json.NewDecoder(resp.Body).Decode(&res); err != nil {
		return nil, err
	}

	embeddings := make([][]float32, len(res.Data))
	for i := range res.Data {
		emb := make([]float32, len(res.Data[i].Embedding))
		for j := range res.Data[i].Embedding {
			emb[j] = float32(res.Data[i].Embedding[j])
		}
		embeddings[i] = emb
	}
	return embeddings, nil
}

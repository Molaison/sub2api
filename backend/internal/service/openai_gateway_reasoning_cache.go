package service

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"time"

	"github.com/Wei-Shaw/sub2api/internal/pkg/apicompat"
	"github.com/Wei-Shaw/sub2api/internal/pkg/logger"
	"go.uber.org/zap"
)

// ReasoningContentBatchCache is optional so alternate GatewayCache backends
// remain compatible. Redis implements it with MGET and a SET pipeline.
type ReasoningContentBatchCache interface {
	GetReasoningContents(context.Context, []string) (map[string]string, error)
	SetReasoningContents(context.Context, map[string]string, time.Duration) error
}

const responsesReasoningCacheBudget = 2 * time.Second

// This state belongs to one request, never to the shared gateway service.
// Cache failures retain the previous best-effort behavior, but are observable.
type responsesReasoningCache struct {
	ctx     context.Context
	cache   GatewayCache
	values  map[string]string
	pending map[string]string
	missing []string
	loaded  bool
}

func (s *OpenAIGatewayService) newResponsesReasoningCache(ctx context.Context, input json.RawMessage) *responsesReasoningCache {
	r := &responsesReasoningCache{ctx: ctx, values: make(map[string]string), pending: make(map[string]string)}
	if s != nil {
		r.cache = s.cache
	}
	var items []json.RawMessage
	if json.Unmarshal(input, &items) != nil {
		return r
	}
	seen := make(map[string]bool)
	for _, raw := range items {
		id, text, ok := apicompat.ExtractResponsesReasoningItem(raw)
		if !ok || id == "" {
			continue
		}
		if text != "" {
			r.values[id] = text
			r.pending[id] = text
		} else if !seen[id] {
			r.missing = append(r.missing, id)
		}
		seen[id] = true
	}
	return r
}

func (r *responsesReasoningCache) lookup(id string) string {
	if text := r.values[id]; text != "" {
		return text
	}
	if r.loaded || r.cache == nil {
		return ""
	}
	r.loaded = true
	ids := make([]string, 0, len(r.missing))
	for _, key := range r.missing {
		if r.values[key] == "" {
			ids = append(ids, key)
		}
	}
	if len(ids) == 0 {
		return ""
	}
	ctx, cancel := context.WithTimeout(r.ctx, responsesReasoningCacheBudget)
	defer cancel()
	start := time.Now()
	var err error
	if batch, ok := r.cache.(ReasoningContentBatchCache); ok {
		var values map[string]string
		values, err = batch.GetReasoningContents(ctx, ids)
		for key, value := range values {
			r.values[key] = value
		}
	} else {
		for _, key := range ids {
			if err = ctx.Err(); err != nil {
				break
			}
			var value string
			value, err = r.cache.GetReasoningContent(ctx, key)
			if errors.Is(err, ErrReasoningContentNotFound) {
				err = nil
				continue
			}
			if err != nil {
				break
			}
			r.values[key] = value
		}
	}
	if err != nil {
		logger.L().Warn("responses reasoning cache lookup failed",
			zap.Int("item_count", len(ids)), zap.Duration("elapsed", time.Since(start)), zap.Error(err))
	}
	return r.values[id]
}

func (r *responsesReasoningCache) collectEvents(events []apicompat.ResponsesStreamEvent) {
	for _, event := range events {
		if event.Type == "response.output_item.done" {
			r.collectItem(event.Item)
		}
	}
}

func (r *responsesReasoningCache) collectItem(item *apicompat.ResponsesOutput) {
	if item == nil || item.Type != "reasoning" || item.ID == "" {
		return
	}
	var parts []string
	for _, part := range item.Summary {
		if text := strings.TrimSpace(part.Text); text != "" {
			parts = append(parts, text)
		}
	}
	if len(parts) != 0 {
		r.pending[item.ID] = strings.Join(parts, "\n")
	}
}

func (r *responsesReasoningCache) flush() {
	if r.cache == nil || len(r.pending) == 0 {
		return
	}
	// Clear before attempting so errors and deferred cleanup do not repeat the
	// entire write budget. No goroutine can outlive this request.
	items := r.pending
	r.pending = make(map[string]string)
	ctx, cancel := context.WithTimeout(context.Background(), responsesReasoningCacheBudget)
	defer cancel()
	start := time.Now()
	var err error
	if batch, ok := r.cache.(ReasoningContentBatchCache); ok {
		err = batch.SetReasoningContents(ctx, items, responsesReasoningCacheTTL)
	} else {
		for id, text := range items {
			if err = ctx.Err(); err != nil {
				break
			}
			if err = r.cache.SetReasoningContent(ctx, id, text, responsesReasoningCacheTTL); err != nil {
				break
			}
		}
	}
	if err != nil {
		logger.L().Warn("responses reasoning cache persistence failed",
			zap.Int("item_count", len(items)), zap.Duration("elapsed", time.Since(start)), zap.Error(err))
	}
}

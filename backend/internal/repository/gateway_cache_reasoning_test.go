package repository

import (
	"context"
	"testing"
	"time"

	"github.com/Wei-Shaw/sub2api/internal/service"
	"github.com/alicebob/miniredis/v2"
	"github.com/redis/go-redis/v9"
	"github.com/stretchr/testify/require"
)

func TestGatewayCacheReasoningContent(t *testing.T) {
	mr := miniredis.RunT(t)
	client := redis.NewClient(&redis.Options{Addr: mr.Addr()})
	cache := NewGatewayCache(client)
	ctx := context.Background()

	// 未命中返回哨兵错误，区别于真实读取失败。
	_, err := cache.GetReasoningContent(ctx, "item_missing")
	require.ErrorIs(t, err, service.ErrReasoningContentNotFound)

	// 写入后可读回。
	require.NoError(t, cache.SetReasoningContent(ctx, "item_abc", "think hard", time.Minute))
	got, err := cache.GetReasoningContent(ctx, "item_abc")
	require.NoError(t, err)
	require.Equal(t, "think hard", got)

	// ttl<=0 时兜底为默认 7 天。
	require.NoError(t, cache.SetReasoningContent(ctx, "item_ttl", "x", 0))
	ttl := mr.TTL(reasoningContentPrefix + "item_ttl")
	require.Greater(t, ttl, 6*24*time.Hour)
	require.LessOrEqual(t, ttl, reasoningContentDefaultTTL)

	// 空 itemID / 空 content 是 no-op（无可缓存内容不算错误）。
	require.NoError(t, cache.SetReasoningContent(ctx, "", "x", time.Minute))
	require.NoError(t, cache.SetReasoningContent(ctx, "item_empty", "", time.Minute))
	_, err = cache.GetReasoningContent(ctx, "item_empty")
	require.ErrorIs(t, err, service.ErrReasoningContentNotFound)
}

func TestGatewayCacheReasoningContentBatch(t *testing.T) {
	mr := miniredis.RunT(t)
	client := redis.NewClient(&redis.Options{Addr: mr.Addr()})
	t.Cleanup(func() { _ = client.Close() })
	cache := NewGatewayCache(client).(service.ReasoningContentBatchCache)
	ctx := context.Background()
	require.NoError(t, cache.SetReasoningContents(ctx, map[string]string{
		"first": "thought one", "second": "thought two", "empty": "", "": "invalid",
	}, time.Minute))
	got, err := cache.GetReasoningContents(ctx, []string{"first", "missing", "second", "", "empty"})
	require.NoError(t, err)
	require.Equal(t, map[string]string{"first": "thought one", "second": "thought two"}, got)
	require.Equal(t, time.Minute, mr.TTL(reasoningContentPrefix+"first"))
	mr.FastForward(time.Minute)
	got, err = cache.GetReasoningContents(ctx, []string{"first", "second"})
	require.NoError(t, err)
	require.Empty(t, got)
	require.NoError(t, cache.SetReasoningContents(ctx, map[string]string{"default": "ttl"}, 0))
	require.Equal(t, reasoningContentDefaultTTL, mr.TTL(reasoningContentPrefix+"default"))
}

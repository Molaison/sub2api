//go:build unit

package service

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
)

type latencyReasoningCache struct {
	reasoningRecordingCache
	beforeSet func()
	getCalls  int
	getError  bool
}

func (c *latencyReasoningCache) SetReasoningContent(ctx context.Context, id, value string, ttl time.Duration) error {
	if c.beforeSet != nil {
		c.beforeSet()
	}
	return c.reasoningRecordingCache.SetReasoningContent(ctx, id, value, ttl)
}

func (c *latencyReasoningCache) GetReasoningContent(ctx context.Context, id string) (string, error) {
	c.getCalls++
	if c.getError {
		<-ctx.Done()
		return "", ctx.Err()
	}
	return c.reasoningRecordingCache.GetReasoningContent(ctx, id)
}

func TestReasoningCacheLatencyDoesNotBlockFirstText(t *testing.T) {
	gin.SetMode(gin.TestMode)
	var history []string
	for i := 0; i < 8; i++ {
		history = append(history, fmt.Sprintf(`{"type":"reasoning","id":"item_%d","encrypted_content":"opaque","summary":[{"type":"summary_text","text":"history"}]}`, i))
	}
	body := []byte(`{"model":"deepseek-reasoner","store":false,"stream":true,"input":[` + strings.Join(history, ",") + `,{"role":"user","content":"hello"}]}`)
	rec := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(rec)
	c.Request = httptest.NewRequest(http.MethodPost, "/v1/responses", bytes.NewReader(body))
	upstream := &httpUpstreamRecorder{resp: &http.Response{
		StatusCode: http.StatusOK,
		Header:     http.Header{"Content-Type": []string{"text/event-stream"}},
		Body: io.NopCloser(strings.NewReader(
			"data: {\"id\":\"chatcmpl_latency\",\"choices\":[{\"index\":0,\"delta\":{\"reasoning_content\":\"plan\"}}]}\n\n" +
				"data: {\"id\":\"chatcmpl_latency\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"answer\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n")),
	}}
	cache := &latencyReasoningCache{}
	var preUpstream, preText, afterTerminal int
	start := time.Now()
	cache.beforeSet = func() {
		if upstream.lastReq == nil {
			preUpstream++
		}
		if !strings.Contains(rec.Body.String(), "event: response.output_text.delta") {
			preText++
		}
		if strings.Contains(rec.Body.String(), "event: response.completed") {
			afterTerminal++
		}
		time.Sleep(25 * time.Millisecond)
	}
	svc := &OpenAIGatewayService{cfg: rawChatCompletionsTestConfig(), httpUpstream: upstream, cache: cache}
	_, err := svc.Forward(context.Background(), c, forceChatResponsesFallbackAccount(), body)
	require.NoError(t, err)
	t.Logf("injected cache delay=25ms/item; writes before upstream=%d before first text=%d; total=%s", preUpstream, preText, time.Since(start))
	require.Zero(t, preUpstream, "history cache refresh must not delay the upstream request")
	require.Zero(t, preText, "cache persistence must follow the flushed first text")
	require.Zero(t, afterTerminal, "next-turn cache must be available before the terminal event")
	require.Len(t, cache.snapshotSets(), 9, "both history and generated reasoning must survive")
	require.Contains(t, rec.Body.String(), "event: response.completed")
}

type latencyBatchReasoningCache struct {
	latencyReasoningCache
	batchGets int
	batchSets int
	requested []string
}

func (c *latencyBatchReasoningCache) GetReasoningContents(_ context.Context, ids []string) (map[string]string, error) {
	c.batchGets++
	c.requested = append([]string(nil), ids...)
	return c.getResp, nil
}

func (c *latencyBatchReasoningCache) SetReasoningContents(ctx context.Context, items map[string]string, ttl time.Duration) error {
	c.batchSets++
	for id, text := range items {
		c.reasoningRecordingCache.SetReasoningContent(ctx, id, text, ttl)
	}
	return nil
}

func TestReasoningCacheBatchesAndPrefersRequestPlaintext(t *testing.T) {
	cache := &latencyBatchReasoningCache{}
	cache.getResp = map[string]string{"enc": "restored"}
	svc := &OpenAIGatewayService{cache: cache}
	r := svc.newResponsesReasoningCache(context.Background(), json.RawMessage(`[
		{"type":"reasoning","id":"plain","summary":[{"type":"summary_text","text":"local"}]},
		{"type":"reasoning","id":"plain","encrypted_content":"opaque"},
		{"type":"reasoning","id":"enc","encrypted_content":"opaque"},
		{"type":"reasoning","id":"enc","encrypted_content":"opaque"},
		{"type":"reasoning","id":"absent","encrypted_content":"opaque"}
	]`))
	require.Equal(t, "local", r.lookup("plain"))
	require.Zero(t, cache.batchGets)
	require.Equal(t, "restored", r.lookup("enc"))
	require.Empty(t, r.lookup("absent"))
	require.Equal(t, "restored", r.lookup("enc"))
	require.Equal(t, 1, cache.batchGets)
	require.ElementsMatch(t, []string{"enc", "absent"}, cache.requested)
	require.Zero(t, cache.getCalls)
	r.flush()
	r.flush()
	require.Equal(t, 1, cache.batchSets)
	require.Equal(t, "local", cache.snapshotSets()["plain"])
}

func TestReasoningCacheReadOutageDoesNotMultiplyTimeouts(t *testing.T) {
	cache := &latencyReasoningCache{getError: true}
	svc := &OpenAIGatewayService{cache: cache}
	ctx, cancel := context.WithTimeout(context.Background(), 40*time.Millisecond)
	defer cancel()
	r := svc.newResponsesReasoningCache(ctx, json.RawMessage(`[
		{"type":"reasoning","id":"one","encrypted_content":"opaque"},
		{"type":"reasoning","id":"two","encrypted_content":"opaque"},
		{"type":"reasoning","id":"three","encrypted_content":"opaque"}
	]`))
	start := time.Now()
	for _, id := range []string{"one", "two", "three"} {
		require.Empty(t, r.lookup(id))
	}
	require.Equal(t, 1, cache.getCalls, "after the outage no other item should consume another timeout")
	require.Less(t, time.Since(start), time.Second)
}

func TestParateraFirstDeltaDoesNotWaitForUpstreamCompletion(t *testing.T) {
	source, writer := io.Pipe()
	upstreamMayFinish := make(chan struct{})
	body := newParateraDeepSeekStreamBody(source, defaultMaxLineSize)
	t.Cleanup(func() { _ = body.Close(); close(upstreamMayFinish) })
	go func() {
		defer writer.Close()
		_, _ = io.WriteString(writer, "data: {\"type\":\"response.reasoning_summary_text.delta\",\"item_id\":\"rs_latency\",\"output_index\":0,\"summary_index\":0,\"delta\":\"first thought\"}\n\n")
		<-upstreamMayFinish
	}()
	received := make(chan string, 1)
	go func() {
		var data strings.Builder
		buffer := make([]byte, 4096)
		for {
			n, err := body.Read(buffer)
			data.Write(buffer[:n])
			if strings.Contains(data.String(), `"delta":"first thought"`) || err != nil {
				received <- data.String()
				return
			}
		}
	}()
	select {
	case data := <-received:
		require.Contains(t, data, `"delta":"first thought"`)
	case <-time.After(time.Second):
		t.Fatal("Paratera first delta was held while the upstream was still streaming")
	}
}

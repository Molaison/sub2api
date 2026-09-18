//go:build unit

package service

import (
	"bytes"
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/Wei-Shaw/sub2api/internal/pkg/openai_compat"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
	"github.com/tidwall/gjson"
)

func storeFalseReplayFixtureBody() []byte {
	return []byte(`{
		"model":"gpt-5.4",
		"store":false,
		"stream":false,
		"tools":[{"type":"function","name":"lookup","parameters":{"type":"object","properties":{"key":{"type":"string"}},"required":["key"]}}],
		"input":[
			{"type":"reasoning","id":"rs_plain","summary":[{"type":"summary_text","text":"lookup the key before answering"}]},
			{"type":"function_call","id":"fc_lookup","call_id":"call_lookup","name":"lookup","arguments":"{\"key\":\"alpha\"}"},
			{"type":"function_call_output","call_id":"call_lookup","output":"alpha result"},
			{"type":"message","role":"user","content":[{"type":"input_text","text":"continue"}]}
		]
	}`)
}

func chatCompletionReplayResponse() string {
	return `{"id":"chatcmpl_cc_replay","object":"chat.completion","model":"gpt-5.4","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":4,"completion_tokens":1,"total_tokens":5}}`
}

func responsesReplayResponse() string {
	return `{"id":"resp_native_replay","object":"response","model":"gpt-5.4","status":"completed","output":[],"usage":{"input_tokens":4,"output_tokens":1,"total_tokens":5}}`
}

func TestForwardResponsesRawCCKeepsStoreFalseReasoningReplay(t *testing.T) {
	gin.SetMode(gin.TestMode)

	body := storeFalseReplayFixtureBody()
	rec := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(rec)
	c.Request = httptest.NewRequest(http.MethodPost, "/v1/responses", bytes.NewReader(body))
	c.Request.Header.Set("Content-Type", "application/json")

	upstream := &httpUpstreamRecorder{resp: &http.Response{
		StatusCode: http.StatusOK,
		Header:     http.Header{"Content-Type": []string{"application/json"}, "x-request-id": []string{"rid_cc_replay"}},
		Body:       io.NopCloser(strings.NewReader(chatCompletionReplayResponse())),
	}}
	svc := &OpenAIGatewayService{
		cfg:          rawChatCompletionsTestConfig(),
		httpUpstream: upstream,
	}

	result, err := svc.Forward(context.Background(), c, forceChatResponsesFallbackAccount(), body)

	require.NoError(t, err)
	require.NotNil(t, result)
	require.Equal(t, "http://upstream.example/v1/chat/completions", upstream.lastReq.URL.String())
	require.False(t, gjson.GetBytes(upstream.lastBody, "input").Exists())
	require.Equal(t, "lookup the key before answering", gjson.GetBytes(upstream.lastBody, "messages.0.reasoning_content").String())
	require.Equal(t, "call_lookup", gjson.GetBytes(upstream.lastBody, "messages.0.tool_calls.0.id").String())
	require.Equal(t, "lookup", gjson.GetBytes(upstream.lastBody, "messages.0.tool_calls.0.function.name").String())
	require.Equal(t, "call_lookup", gjson.GetBytes(upstream.lastBody, "messages.1.tool_call_id").String())
	require.Equal(t, "alpha result", gjson.GetBytes(upstream.lastBody, "messages.1.content").String())
}

func TestForwardResponsesNativeStoreFalseReasoningCleanupStillApplies(t *testing.T) {
	tests := []struct {
		name  string
		extra map[string]any
	}{
		{
			name: "explicit responses support",
			extra: map[string]any{
				openai_compat.ExtraKeyResponsesMode:      string(openai_compat.ResponsesSupportModeAuto),
				openai_compat.ExtraKeyResponsesSupported: true,
			},
		},
		{
			name:  "unknown defaults to responses",
			extra: map[string]any{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gin.SetMode(gin.TestMode)

			body := storeFalseReplayFixtureBody()
			rec := httptest.NewRecorder()
			c, _ := gin.CreateTestContext(rec)
			c.Request = httptest.NewRequest(http.MethodPost, "/v1/responses", bytes.NewReader(body))
			c.Request.Header.Set("Content-Type", "application/json")

			upstream := &httpUpstreamRecorder{resp: &http.Response{
				StatusCode: http.StatusOK,
				Header:     http.Header{"Content-Type": []string{"application/json"}, "x-request-id": []string{"rid_native_replay"}},
				Body:       io.NopCloser(strings.NewReader(responsesReplayResponse())),
			}}
			svc := &OpenAIGatewayService{
				cfg:          rawChatCompletionsTestConfig(),
				httpUpstream: upstream,
			}
			account := rawChatCompletionsTestAccount()
			account.Extra = tt.extra

			result, err := svc.Forward(context.Background(), c, account, body)

			require.NoError(t, err)
			require.NotNil(t, result)
			require.Equal(t, "http://upstream.example/v1/responses", upstream.lastReq.URL.String())
			require.False(t, gjson.GetBytes(upstream.lastBody, "messages").Exists())
			require.Equal(t, "function_call", gjson.GetBytes(upstream.lastBody, "input.0.type").String())
			require.Equal(t, "call_lookup", gjson.GetBytes(upstream.lastBody, "input.0.call_id").String())
			require.Equal(t, "function_call_output", gjson.GetBytes(upstream.lastBody, "input.1.type").String())
			require.Equal(t, "call_lookup", gjson.GetBytes(upstream.lastBody, "input.1.call_id").String())
			require.NotContains(t, gjson.GetBytes(upstream.lastBody, "input").Raw, `"type":"reasoning"`)
		})
	}
}

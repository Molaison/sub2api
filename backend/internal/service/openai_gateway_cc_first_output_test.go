//go:build unit

package service

import (
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/Wei-Shaw/sub2api/internal/pkg/apicompat"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
)

func firstOutputStringPtr(value string) *string {
	return &value
}

func TestChatChunkStartsResponsesOutput(t *testing.T) {
	tests := []struct {
		name  string
		chunk *apicompat.ChatCompletionsChunk
		want  bool
	}{
		{
			name: "empty reasoning content is not output",
			chunk: &apicompat.ChatCompletionsChunk{Choices: []apicompat.ChatChunkChoice{{
				Delta: apicompat.ChatDelta{ReasoningContent: firstOutputStringPtr("")},
			}}},
			want: false,
		},
		{
			name: "non-empty reasoning alias is output",
			chunk: &apicompat.ChatCompletionsChunk{Choices: []apicompat.ChatChunkChoice{{
				Delta: apicompat.ChatDelta{Reasoning: firstOutputStringPtr("plan")},
			}}},
			want: true,
		},
		{
			name: "empty reasoning alias is not output",
			chunk: &apicompat.ChatCompletionsChunk{Choices: []apicompat.ChatChunkChoice{{
				Delta: apicompat.ChatDelta{Reasoning: firstOutputStringPtr("")},
			}}},
			want: false,
		},
		{
			name: "empty reasoning content keeps precedence over alias",
			chunk: &apicompat.ChatCompletionsChunk{Choices: []apicompat.ChatChunkChoice{{
				Delta: apicompat.ChatDelta{
					ReasoningContent: firstOutputStringPtr(""),
					Reasoning:        firstOutputStringPtr("alias must not win"),
				},
			}}},
			want: false,
		},
		{
			name: "non-empty content is output",
			chunk: &apicompat.ChatCompletionsChunk{Choices: []apicompat.ChatChunkChoice{{
				Delta: apicompat.ChatDelta{Content: firstOutputStringPtr("answer")},
			}}},
			want: true,
		},
		{
			name: "empty content is not output",
			chunk: &apicompat.ChatCompletionsChunk{Choices: []apicompat.ChatChunkChoice{{
				Delta: apicompat.ChatDelta{Content: firstOutputStringPtr("")},
			}}},
			want: false,
		},
		{
			name: "tool call is output",
			chunk: &apicompat.ChatCompletionsChunk{Choices: []apicompat.ChatChunkChoice{{
				Delta: apicompat.ChatDelta{ToolCalls: []apicompat.ChatToolCall{{
					Function: apicompat.ChatFunctionCall{Name: "lookup"},
				}}},
			}}},
			want: true,
		},
		{
			name: "usage-only chunk is not output",
			chunk: &apicompat.ChatCompletionsChunk{
				Choices: []apicompat.ChatChunkChoice{},
				Usage:   &apicompat.ChatUsage{PromptTokens: 1, CompletionTokens: 1, TotalTokens: 2},
			},
			want: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			require.Equal(t, tt.want, chatChunkStartsResponsesOutput(tt.chunk))
		})
	}
}

func TestScanCCStreamFirstOutputIgnoresEmptyReasoning(t *testing.T) {
	tests := []struct {
		name string
		sse  string
		want bool
	}{
		{
			name: "empty reasoning and usage-only do not start output",
			sse: strings.Join([]string{
				`data: {"id":"chatcmpl_first","choices":[{"index":0,"delta":{"role":"assistant","content":null,"reasoning_content":""},"finish_reason":null}]}`,
				"",
				`data: {"id":"chatcmpl_first","choices":[],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`,
				"",
				"data: [DONE]",
				"",
			}, "\n"),
			want: false,
		},
		{
			name: "empty reasoning content does not fall back to alias",
			sse: strings.Join([]string{
				`data: {"id":"chatcmpl_first","choices":[{"index":0,"delta":{"reasoning_content":"","reasoning":"alias"},"finish_reason":null}]}`,
				"",
				"data: [DONE]",
				"",
			}, "\n"),
			want: false,
		},
		{
			name: "non-empty reasoning alias starts output",
			sse: strings.Join([]string{
				`data: {"id":"chatcmpl_first","choices":[{"index":0,"delta":{"reasoning":"plan"},"finish_reason":null}]}`,
				"",
				"data: [DONE]",
				"",
			}, "\n"),
			want: true,
		},
		{
			name: "non-empty content starts output",
			sse: strings.Join([]string{
				`data: {"id":"chatcmpl_first","choices":[{"index":0,"delta":{"content":"answer"},"finish_reason":null}]}`,
				"",
				"data: [DONE]",
				"",
			}, "\n"),
			want: true,
		},
		{
			name: "tool call starts output",
			sse: strings.Join([]string{
				`data: {"id":"chatcmpl_first","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"lookup","arguments":"{}"}}]},"finish_reason":null}]}`,
				"",
				"data: [DONE]",
				"",
			}, "\n"),
			want: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gin.SetMode(gin.TestMode)
			rec := httptest.NewRecorder()
			c, _ := gin.CreateTestContext(rec)
			resp := &http.Response{
				StatusCode: http.StatusOK,
				Header:     http.Header{"Content-Type": []string{"text/event-stream"}},
				Body:       io.NopCloser(strings.NewReader(tt.sse)),
			}
			svc := &OpenAIGatewayService{}

			state := svc.scanCCStream(c, resp, "test", "rid_first_output", time.Now().Add(-time.Millisecond), func(*apicompat.ChatCompletionsChunk) {})

			require.NoError(t, state.Err)
			if tt.want {
				require.NotNil(t, state.FirstTokenMs)
			} else {
				require.Nil(t, state.FirstTokenMs)
			}
		})
	}
}

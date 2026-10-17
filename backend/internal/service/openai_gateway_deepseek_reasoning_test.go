package service

import (
	"bytes"
	"encoding/json"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
	"github.com/tidwall/gjson"
)

func deepSeekNativeResponsesTestAccount() *Account {
	return &Account{
		Platform: PlatformDeepseek,
		Type:     AccountTypeAPIKey,
		Credentials: map[string]any{
			"api_protocol": APIProtocolResponses,
		},
	}
}

func deepSeekReasoningTestContext(userAgent string) *gin.Context {
	c, _ := gin.CreateTestContext(httptest.NewRecorder())
	c.Request = httptest.NewRequest("POST", "/v1/responses", nil)
	c.Request.Header.Set("User-Agent", userAgent)
	return c
}

func TestDeepSeekCodexReasoningCompatibilityScope(t *testing.T) {
	svc := &OpenAIGatewayService{}
	body := []byte(`{"output":[{"type":"reasoning","summary":[{"type":"summary_text","text":"thinking"}]}]}`)
	deepSeek := deepSeekNativeResponsesTestAccount()

	nonCodex := svc.decorateDeepSeekCodexReasoningResponse(deepSeekReasoningTestContext("curl/8.0"), deepSeek, body)
	require.True(t, bytes.Equal(body, nonCodex))

	openAIAccount := &Account{Platform: PlatformOpenAI, Type: AccountTypeAPIKey}
	nonDeepSeek := svc.decorateDeepSeekCodexReasoningResponse(deepSeekReasoningTestContext("codex_cli_rs/0.154.0"), openAIAccount, body)
	require.True(t, bytes.Equal(body, nonDeepSeek))

	codexDeepSeek := svc.decorateDeepSeekCodexReasoningResponse(deepSeekReasoningTestContext("codex_cli_rs/0.154.0"), deepSeek, body)
	require.Equal(t, deepSeekCodexReasoningHeader+"thinking", gjson.GetBytes(codexDeepSeek, "output.0.summary.0.text").String())
}

func TestDeepSeekCodexReasoningStreamEvents(t *testing.T) {
	decorator := newDeepSeekCodexReasoningStreamDecorator(true)
	require.NotNil(t, decorator)

	firstDelta := []byte(`{"type":"response.reasoning_summary_text.delta","item_id":"rs_1","output_index":0,"summary_index":0,"delta":"first"}`)
	got, changed := decorator.Decorate(firstDelta, "response.reasoning_summary_text.delta")
	require.True(t, changed)
	require.Equal(t, deepSeekCodexReasoningHeader+"first", gjson.GetBytes(got, "delta").String())

	secondDelta := []byte(`{"type":"response.reasoning_summary_text.delta","item_id":"rs_1","output_index":0,"summary_index":0,"delta":" second"}`)
	got, changed = decorator.Decorate(secondDelta, "response.reasoning_summary_text.delta")
	require.False(t, changed)
	require.True(t, bytes.Equal(secondDelta, got))

	nextSummary := []byte(`{"type":"response.reasoning_summary_text.delta","item_id":"rs_1","output_index":0,"summary_index":1,"delta":"next"}`)
	got, changed = decorator.Decorate(nextSummary, "response.reasoning_summary_text.delta")
	require.True(t, changed)
	require.Equal(t, deepSeekCodexReasoningHeader+"next", gjson.GetBytes(got, "delta").String())

	tests := []struct {
		name      string
		eventType string
		payload   string
		path      string
	}{
		{
			name:      "text done",
			eventType: "response.reasoning_summary_text.done",
			payload:   `{"type":"response.reasoning_summary_text.done","text":"thinking"}`,
			path:      "text",
		},
		{
			name:      "part done",
			eventType: "response.reasoning_summary_part.done",
			payload:   `{"type":"response.reasoning_summary_part.done","part":{"type":"summary_text","text":"thinking"}}`,
			path:      "part.text",
		},
		{
			name:      "output item done",
			eventType: "response.output_item.done",
			payload:   `{"type":"response.output_item.done","item":{"type":"reasoning","summary":[{"type":"summary_text","text":"thinking"}]}}`,
			path:      "item.summary.0.text",
		},
		{
			name:      "response completed",
			eventType: "response.completed",
			payload:   `{"type":"response.completed","response":{"output":[{"type":"reasoning","summary":[{"type":"summary_text","text":"thinking"}]}]}}`,
			path:      "response.output.0.summary.0.text",
		},
		{
			name:      "response done",
			eventType: "response.done",
			payload:   `{"type":"response.done","response":{"output":[{"type":"reasoning","summary":[{"type":"summary_text","text":"thinking"}]}]}}`,
			path:      "response.output.0.summary.0.text",
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			decorated, eventChanged := decorator.Decorate([]byte(test.payload), test.eventType)
			require.True(t, eventChanged)
			require.Equal(t, deepSeekCodexReasoningHeader+"thinking", gjson.GetBytes(decorated, test.path).String())
		})
	}

	partAdded := []byte(`{"type":"response.reasoning_summary_part.added","part":{"type":"summary_text","text":""}}`)
	got, changed = decorator.Decorate(partAdded, "response.reasoning_summary_part.added")
	require.False(t, changed)
	require.True(t, bytes.Equal(partAdded, got))
}

func TestDeepSeekCodexReasoningResponseIsIdempotent(t *testing.T) {
	body := []byte(`{"id":"resp_1","opaque_integer":9007199254740993123456789,"output":[{"type":"reasoning","summary":[{"type":"summary_text","text":"thinking"}]},{"type":"message","content":[{"type":"output_text","text":"answer"}]}]}`)
	decorated, changed := decorateDeepSeekReasoningItemsAtPath(body, "output")
	require.True(t, changed)
	require.Equal(t, deepSeekCodexReasoningHeader+"thinking", gjson.GetBytes(decorated, "output.0.summary.0.text").String())
	require.Contains(t, string(decorated), `9007199254740993123456789`)

	decoratedAgain, changedAgain := decorateDeepSeekReasoningItemsAtPath(decorated, "output")
	require.False(t, changedAgain)
	require.True(t, bytes.Equal(decorated, decoratedAgain))
}

func TestStripDeepSeekCodexReasoningHeaders(t *testing.T) {
	body := []byte(`{"opaque_integer":9007199254740993123456789,"input":[{"type":"reasoning","summary":[{"type":"summary_text","text":"**DeepSeek reasoning**\n\nmodel thought"},{"type":"summary_text","text":"**User title**\n\nkeep me"}]},{"type":"message","role":"user","content":"**DeepSeek reasoning**\n\nnot a reasoning summary"}]}`)

	stripped, changed, err := stripDeepSeekCodexReasoningHeaders(body)
	require.NoError(t, err)
	require.True(t, changed)
	require.Equal(t, "model thought", gjson.GetBytes(stripped, "input.0.summary.0.text").String())
	require.Equal(t, "**User title**\n\nkeep me", gjson.GetBytes(stripped, "input.0.summary.1.text").String())
	require.Equal(t, "**DeepSeek reasoning**\n\nnot a reasoning summary", gjson.GetBytes(stripped, "input.1.content").String())
	require.Contains(t, string(stripped), `9007199254740993123456789`)
}

func TestDeepSeekCodexReasoningInjectStripRoundTrip(t *testing.T) {
	originalText := "inspect the tool result"
	response := []byte(`{"output":[{"type":"reasoning","summary":[{"type":"summary_text","text":"inspect the tool result"}]}]}`)
	decorated, changed := decorateDeepSeekReasoningItemsAtPath(response, "output")
	require.True(t, changed)

	request := map[string]any{
		"input": []any{
			map[string]any{
				"type": "reasoning",
				"summary": []any{
					map[string]any{"type": "summary_text", "text": gjson.GetBytes(decorated, "output.0.summary.0.text").String()},
				},
			},
		},
	}
	replayed, err := json.Marshal(request)
	require.NoError(t, err)
	stripped, strippedChanged, err := stripDeepSeekCodexReasoningHeaders(replayed)
	require.NoError(t, err)
	require.True(t, strippedChanged)
	require.Equal(t, originalText, gjson.GetBytes(stripped, "input.0.summary.0.text").String())
	require.Equal(t, 1, strings.Count(gjson.GetBytes(decorated, "output.0.summary.0.text").String(), deepSeekCodexReasoningHeader))
}

package service

import (
	"bytes"
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
	"github.com/tidwall/gjson"
)

// brokenParateraStream is the exact event sequence Paratera's
// DeepSeek-V4.1-Flash endpoint emits when a turn produces reasoning before the
// assistant text: the message item is never announced, so the client sees a
// text delta with no active item.
const brokenParateraStream = `data: {"type":"response.created","response":{"id":"resp_live"},"model":"DeepSeek-V4.1-Flash"}

data: {"type":"response.in_progress","response":{"id":"resp_live"},"model":"DeepSeek-V4.1-Flash"}

data: {"type":"response.output_item.added","output_index":0,"item":{"id":"rs_live","type":"reasoning","status":"in_progress"},"model":"DeepSeek-V4.1-Flash"}

data: {"type":"response.reasoning_summary_text.delta","item_id":"rs_live","output_index":0,"summary_index":0,"delta":"thinking","model":"DeepSeek-V4.1-Flash"}

data: {"type":"response.reasoning_summary_text.done","item_id":"rs_live","output_index":0,"summary_index":0,"text":"thinking","model":"DeepSeek-V4.1-Flash"}

data: {"type":"response.output_item.done","output_index":0,"item":{"id":"rs_live","type":"reasoning","status":"completed","summary":[{"type":"summary_text","text":"thinking"}]},"model":"DeepSeek-V4.1-Flash"}

data: {"type":"response.output_text.delta","item_id":"msg_live","output_index":0,"content_index":0,"delta":"你好呀","model":"DeepSeek-V4.1-Flash"}

data: {"type":"response.output_text.done","item_id":"msg_live","output_index":0,"content_index":0,"text":"你好呀","model":"DeepSeek-V4.1-Flash"}

data: {"type":"response.output_item.done","output_index":0,"item":{"id":"msg_live","type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":"你好呀"}]},"model":"DeepSeek-V4.1-Flash"}

data: {"type":"response.completed","response":{"id":"resp_live","status":"completed","output":[{"id":"msg_live","type":"message","role":"assistant","content":[{"type":"output_text","text":"你好呀"}]}],"usage":{"input_tokens":10,"output_tokens":2}}}

`

func parateraStreamingTestAccount() *Account {
	return &Account{
		ID:       9001,
		Platform: PlatformOpenAI,
		Type:     AccountTypeAPIKey,
		Credentials: map[string]any{
			"api_key":  "test-key",
			"base_url": parateraDeepSeekResponsesBaseURL,
		},
	}
}

// A client consuming the streamed events must always see the message item
// opened before its text. Before the repair this assertion fails for the
// reasoning-then-text shape Paratera produces.
func TestParateraDeepSeekPassthroughOpensMessageItemBeforeText(t *testing.T) {
	gin.SetMode(gin.TestMode)
	body := []byte(`{"model":"DeepSeek-V4.1-Flash","stream":true,"input":"只回复：你好呀"}`)
	recorder := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(recorder)
	c.Request = httptest.NewRequest(http.MethodPost, "/v1/responses", bytes.NewReader(body))
	c.Request.Header.Set("originator", "codex_cli_rs")
	c.Request.Header.Set("User-Agent", "codex_cli_rs/0.154.0")

	upstream := &httpUpstreamRecorder{resp: &http.Response{
		StatusCode: http.StatusOK,
		Header:     http.Header{"Content-Type": []string{"text/event-stream"}},
		Body:       io.NopCloser(strings.NewReader(brokenParateraStream)),
	}}
	svc := openAIClientToolsTestService(upstream)

	_, err := svc.forwardOpenAIPassthrough(context.Background(), c, parateraStreamingTestAccount(), body, body, "DeepSeek-V4.1-Flash", false, nil, true, time.Now())
	require.NoError(t, err)

	stream := recorder.Body.String()
	require.Contains(t, stream, "response.output_text.delta", "assistant text must reach the client")

	types := eventTypes(ssePayloads(t, stream))
	deltaIndex := indexOfEventType(types, "response.output_text.delta")
	require.Greater(t, deltaIndex, 1)
	require.Equal(t, "response.content_part.added", types[deltaIndex-1])
	require.Equal(t, "response.output_item.added", types[deltaIndex-2])
	require.Equal(t, "msg_live", gjsonGetString(ssePayloads(t, stream)[deltaIndex-2], "item.id"))

	// Reasoning must survive untouched alongside the repaired message item.
	require.Equal(t, 1, countEventType(types, "response.reasoning_summary_text.delta"))
	require.Equal(t, "response.completed", types[len(types)-1])
}

// A non-Paratera account must not have its stream rewritten.
func TestParateraDeepSeekRepairNotAppliedToOtherAccounts(t *testing.T) {
	gin.SetMode(gin.TestMode)
	body := []byte(`{"model":"DeepSeek-V4.1-Flash","stream":true,"input":"hi"}`)
	recorder := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(recorder)
	c.Request = httptest.NewRequest(http.MethodPost, "/v1/responses", bytes.NewReader(body))

	upstream := &httpUpstreamRecorder{resp: &http.Response{
		StatusCode: http.StatusOK,
		Header:     http.Header{"Content-Type": []string{"text/event-stream"}},
		Body:       io.NopCloser(strings.NewReader(brokenParateraStream)),
	}}
	svc := openAIClientToolsTestService(upstream)
	account := &Account{ID: 9002, Platform: PlatformOpenAI, Type: AccountTypeAPIKey,
		Credentials: map[string]any{"api_key": "test-key", "base_url": "https://api.openai.com/v1"}}

	_, err := svc.forwardOpenAIPassthrough(context.Background(), c, account, body, body, "DeepSeek-V4.1-Flash", false, nil, true, time.Now())
	require.NoError(t, err)

	types := eventTypes(ssePayloads(t, recorder.Body.String()))
	require.Equal(t, 0, countEventType(types, "response.content_part.added"))
}

func gjsonGetString(payload, path string) string {
	return gjson.GetBytes([]byte(payload), path).String()
}

// ssePayloads returns the JSON payload of every data: line in an SSE stream.
func ssePayloads(t *testing.T, stream string) []string {
	t.Helper()
	var payloads []string
	for _, line := range strings.Split(stream, "\n") {
		if data, ok := extractOpenAISSEDataLine(line); ok {
			payloads = append(payloads, data)
		}
	}
	return payloads
}

func eventTypes(payloads []string) []string {
	types := make([]string, 0, len(payloads))
	for _, payload := range payloads {
		types = append(types, gjsonGetString(payload, "type"))
	}
	return types
}

func runParateraRepair(t *testing.T, upstream string) string {
	t.Helper()
	body := newParateraDeepSeekStreamBody(io.NopCloser(strings.NewReader(upstream)), defaultMaxLineSize)
	repaired, err := io.ReadAll(body)
	require.NoError(t, err)
	require.NoError(t, body.Close())
	return string(repaired)
}

// Paratera omits the message item's output_item.added whenever the turn also
// carries reasoning: the stream jumps from the reasoning item straight to
// output_text.delta. Codex then drops the delta ("OutputTextDelta without
// active item"), so the repair must announce the item first.
func TestParateraDeepSeekRepairAnnouncesMessageBeforeTextDelta(t *testing.T) {
	const upstream = `data: {"type":"response.created","response":{"id":"resp_1"},"model":"DeepSeek-V4.1-Flash"}

data: {"type":"response.in_progress","response":{"id":"resp_1"},"model":"DeepSeek-V4.1-Flash"}

data: {"type":"response.output_item.added","output_index":0,"item":{"id":"rs_1","type":"reasoning","status":"in_progress"},"model":"DeepSeek-V4.1-Flash"}

data: {"type":"response.reasoning_summary_text.delta","item_id":"rs_1","output_index":0,"summary_index":0,"delta":"think","model":"DeepSeek-V4.1-Flash"}

data: {"type":"response.reasoning_summary_text.done","item_id":"rs_1","output_index":0,"summary_index":0,"text":"think","model":"DeepSeek-V4.1-Flash"}

data: {"type":"response.output_item.done","output_index":0,"item":{"id":"rs_1","type":"reasoning","status":"completed","summary":[{"type":"summary_text","text":"think"}]},"model":"DeepSeek-V4.1-Flash"}

data: {"type":"response.output_text.delta","item_id":"msg_1","output_index":0,"content_index":0,"delta":"你好呀","model":"DeepSeek-V4.1-Flash"}

data: {"type":"response.output_text.done","item_id":"msg_1","output_index":0,"content_index":0,"text":"你好呀","model":"DeepSeek-V4.1-Flash"}

data: {"type":"response.output_item.done","output_index":0,"item":{"id":"msg_1","type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":"你好呀"}]},"model":"DeepSeek-V4.1-Flash"}

data: {"type":"response.completed","response":{"id":"resp_1","output":[]}}

`
	payloads := ssePayloads(t, runParateraRepair(t, upstream))
	types := eventTypes(payloads)

	// The synthesized announcements must land right before the text delta.
	deltaIndex := indexOfEventType(types, "response.output_text.delta")
	require.Greater(t, deltaIndex, 0, "text delta must survive the repair")
	require.Equal(t, "response.content_part.added", types[deltaIndex-1])
	require.Equal(t, "response.output_item.added", types[deltaIndex-2])

	messageAdded := payloads[deltaIndex-2]
	require.Equal(t, "msg_1", gjsonGetString(messageAdded, "item.id"))
	require.Equal(t, "message", gjsonGetString(messageAdded, "item.type"))
	require.Equal(t, "assistant", gjsonGetString(messageAdded, "item.role"))
	require.Equal(t, "in_progress", gjsonGetString(messageAdded, "item.status"))
	require.Equal(t, "DeepSeek-V4.1-Flash", gjsonGetString(messageAdded, "model"))

	partAdded := payloads[deltaIndex-1]
	require.Equal(t, "msg_1", gjsonGetString(partAdded, "item_id"))
	require.Equal(t, "output_text", gjsonGetString(partAdded, "part.type"))

	// Every event the client relies on is preserved in order.
	require.Equal(t, []string{
		"response.created",
		"response.in_progress",
		"response.output_item.added",
		"response.reasoning_summary_text.delta",
		"response.reasoning_summary_text.done",
		"response.output_item.done",
		"response.output_item.added",
		"response.content_part.added",
		"response.output_text.delta",
		"response.output_text.done",
		"response.output_item.done",
		"response.completed",
	}, types)
}

// A healthy upstream stream must pass through byte-for-byte: no duplicate
// announcements, no dropped frames.
func TestParateraDeepSeekRepairLeavesAnnouncedStreamUntouched(t *testing.T) {
	const upstream = `data: {"type":"response.created","response":{"id":"resp_2"},"model":"DeepSeek-V4.1-Flash"}

data: {"type":"response.output_item.added","output_index":0,"item":{"id":"msg_2","type":"message","role":"assistant","status":"in_progress","content":[]},"model":"DeepSeek-V4.1-Flash"}

data: {"type":"response.content_part.added","item_id":"msg_2","output_index":0,"content_index":0,"part":{"type":"output_text","text":""},"model":"DeepSeek-V4.1-Flash"}

data: {"type":"response.output_text.delta","item_id":"msg_2","output_index":0,"content_index":0,"delta":"hi","model":"DeepSeek-V4.1-Flash"}

data: {"type":"response.output_item.done","output_index":0,"item":{"id":"msg_2","type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":"hi"}]},"model":"DeepSeek-V4.1-Flash"}

data: {"type":"response.completed","response":{"id":"resp_2","output":[]}}

`
	require.Equal(t, upstream, runParateraRepair(t, upstream))
}

// Reasoning-only items and tool calls are unrelated to the message lifecycle
// and must not attract synthesized announcements.
func TestParateraDeepSeekRepairIgnoresNonMessageItems(t *testing.T) {
	const upstream = `data: {"type":"response.output_item.added","output_index":0,"item":{"id":"call_1","type":"function_call","status":"in_progress"},"model":"DeepSeek-V4.1-Flash"}

data: {"type":"response.function_call_arguments.done","item_id":"call_1","output_index":0,"arguments":"{}","model":"DeepSeek-V4.1-Flash"}

data: {"type":"response.output_item.done","output_index":0,"item":{"id":"call_1","type":"function_call","status":"completed"},"model":"DeepSeek-V4.1-Flash"}

`
	require.Equal(t, upstream, runParateraRepair(t, upstream))
}

// Paratera repeats the same message item id on every text event; only the
// first one may trigger an announcement.
func TestParateraDeepSeekRepairAnnouncesMessageOnlyOnce(t *testing.T) {
	const upstream = `data: {"type":"response.output_text.delta","item_id":"msg_3","output_index":0,"content_index":0,"delta":"a","model":"DeepSeek-V4.1-Flash"}

data: {"type":"response.output_text.delta","item_id":"msg_3","output_index":0,"content_index":0,"delta":"b","model":"DeepSeek-V4.1-Flash"}

data: {"type":"response.output_text.done","item_id":"msg_3","output_index":0,"content_index":0,"text":"ab","model":"DeepSeek-V4.1-Flash"}

data: {"type":"response.output_item.done","output_index":0,"item":{"id":"msg_3","type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":"ab"}]},"model":"DeepSeek-V4.1-Flash"}

`
	types := eventTypes(ssePayloads(t, runParateraRepair(t, upstream)))
	require.Equal(t, 1, countEventType(types, "response.output_item.added"))
	require.Equal(t, 1, countEventType(types, "response.content_part.added"))
}

// Frames dropped by the repair must not leave an orphan blank line behind.
func TestParateraDeepSeekRepairRepairsRepeatedAnnouncement(t *testing.T) {
	const upstream = `data: {"type":"response.output_text.delta","item_id":"msg_4","output_index":0,"content_index":0,"delta":"x","model":"DeepSeek-V4.1-Flash"}

data: {"type":"response.output_item.added","output_index":0,"item":{"id":"msg_4","type":"message","role":"assistant","status":"in_progress","content":[]},"model":"DeepSeek-V4.1-Flash"}

`
	repaired := runParateraRepair(t, upstream)
	require.NotContains(t, repaired, "\n\n\n")
	types := eventTypes(ssePayloads(t, repaired))
	require.Equal(t, 1, countEventType(types, "response.output_item.added"))
}

// Paratera omits summary_index on reasoning events, and Codex's parser drops
// the event entirely when it is missing (it destructures delta and
// summary_index together). The repair must add it so reasoning streams.
func TestParateraDeepSeekRepairAddsMissingSummaryIndex(t *testing.T) {
	const upstream = `data: {"type":"response.output_item.added","output_index":0,"item":{"id":"rs_1","type":"reasoning","status":"in_progress"},"model":"DeepSeek-V4.1-Flash"}

data: {"type":"response.reasoning_summary_text.delta","item_id":"rs_1","output_index":0,"delta":"think","model":"DeepSeek-V4.1-Flash"}

data: {"type":"response.reasoning_summary_text.done","item_id":"rs_1","output_index":0,"text":"think","model":"DeepSeek-V4.1-Flash"}

data: {"type":"response.output_item.done","output_index":0,"item":{"id":"rs_1","type":"reasoning","status":"completed","summary":[{"type":"summary_text","text":"think"}]},"model":"DeepSeek-V4.1-Flash"}

`
	payloads := ssePayloads(t, runParateraRepair(t, upstream))

	var delta string
	for _, payload := range payloads {
		if gjsonGetString(payload, "type") == "response.reasoning_summary_text.delta" {
			delta = payload
		}
	}
	require.NotEmpty(t, delta, "reasoning delta must survive the repair")
	require.True(t, gjson.Get(delta, "summary_index").Exists(),
		"reasoning delta must carry summary_index or Codex discards it")
	require.Equal(t, int64(0), gjson.Get(delta, "summary_index").Int())
	// The delta text itself must be untouched.
	require.Equal(t, "think", gjsonGetString(delta, "delta"))
}

// A stream that already carries summary_index must pass through byte-for-byte.
func TestParateraDeepSeekRepairKeepsExistingSummaryIndex(t *testing.T) {
	const upstream = `data: {"type":"response.reasoning_summary_text.delta","item_id":"rs_1","output_index":0,"summary_index":0,"delta":"think","model":"DeepSeek-V4.1-Flash"}

`
	require.Equal(t, upstream, runParateraRepair(t, upstream))
}

func TestShouldRepairParateraDeepSeekStreamScope(t *testing.T) {
	paratera := &Account{
		Platform:    PlatformOpenAI,
		Type:        AccountTypeAPIKey,
		Credentials: map[string]any{"base_url": parateraDeepSeekResponsesBaseURL},
	}
	require.True(t, shouldRepairParateraDeepSeekStream(paratera, parateraDeepSeekResponsesModel))

	// Same endpoint, different model: the defect is specific to this model.
	require.False(t, shouldRepairParateraDeepSeekStream(paratera, "gpt-5.6-sol"))

	otherBase := &Account{
		Platform:    PlatformOpenAI,
		Type:        AccountTypeAPIKey,
		Credentials: map[string]any{"base_url": "https://api.openai.com/v1"},
	}
	require.False(t, shouldRepairParateraDeepSeekStream(otherBase, parateraDeepSeekResponsesModel))

	// OAuth accounts never hit the Paratera API-key endpoint.
	require.False(t, shouldRepairParateraDeepSeekStream(&Account{Platform: PlatformOpenAI, Type: AccountTypeOAuth}, parateraDeepSeekResponsesModel))
	require.False(t, shouldRepairParateraDeepSeekStream(nil, parateraDeepSeekResponsesModel))
}

func indexOfEventType(types []string, want string) int {
	for i, eventType := range types {
		if eventType == want {
			return i
		}
	}
	return -1
}

func countEventType(types []string, want string) int {
	count := 0
	for _, eventType := range types {
		if eventType == want {
			count++
		}
	}
	return count
}

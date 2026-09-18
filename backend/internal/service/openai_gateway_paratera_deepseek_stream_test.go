package service

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
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
		"response.reasoning_summary_part.added",
		"response.reasoning_summary_text.delta",
		"response.reasoning_summary_text.done",
		"response.reasoning_summary_part.done",
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
	payloads := ssePayloads(t, runParateraRepair(t, upstream))
	// The repair must not synthesize a second announcement for an item the
	// upstream already announced; it only canonicalizes the lifecycle.
	added := 0
	for _, payload := range payloads {
		if gjsonGetString(payload, "type") == "response.output_item.added" {
			added++
			require.Equal(t, "msg_2", gjsonGetString(payload, "item.id"))
		}
	}
	require.Equal(t, 1, added, "already-announced item must not gain a duplicate announcement")
	terminal := payloads[len(payloads)-1]
	require.Equal(t, "response.completed", gjsonGetString(terminal, "type"))
	require.Equal(t, "msg_2", gjsonGetString(terminal, "response.output.0.id"),
		"terminal output must match the delivered item")
}

// The captured production sample reuses output_index 0 for both the reasoning
// and message items and renames the message id in the terminal snapshot. That is
// recoverable renumbering/id drift: indices must be canonicalized and the
// terminal output rebuilt from delivered items, while the response still
// completes normally.
func TestParateraDeepSeekRepairCanonicalIndicesAndTerminalConsistency(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join("testdata", "paratera_streams", "gateway_smoke_para_shared_index.sse"))
	require.NoError(t, err)
	payloads := ssePayloads(t, runParateraRepair(t, string(raw)))

	var addedIdx, doneIdx []int64
	var doneItems []map[string]any
	for _, payload := range payloads {
		switch gjsonGetString(payload, "type") {
		case "response.output_item.added":
			addedIdx = append(addedIdx, gjson.GetBytes([]byte(payload), "output_index").Int())
		case "response.output_item.done":
			doneIdx = append(doneIdx, gjson.GetBytes([]byte(payload), "output_index").Int())
			var item map[string]any
			require.NoError(t, json.Unmarshal([]byte(gjson.GetBytes([]byte(payload), "item").Raw), &item))
			doneItems = append(doneItems, item)
		}
	}
	require.Equal(t, []int64{0, 1}, addedIdx, "added output_index must be canonical and contiguous")
	require.Equal(t, []int64{0, 1}, doneIdx, "done output_index must be canonical and contiguous")

	var terminal string
	for _, payload := range payloads {
		switch gjsonGetString(payload, "type") {
		case "response.completed", "response.incomplete", "response.failed":
			terminal = payload
		}
	}
	require.NotEmpty(t, terminal, "a terminal event must be present")
	require.Equal(t, "response.completed", gjsonGetString(terminal, "type"),
		"recoverable renumbering/id drift must complete normally, not fail")
	var terminalOutput []map[string]any
	require.NoError(t, json.Unmarshal([]byte(gjson.GetBytes([]byte(terminal), "response.output").Raw), &terminalOutput))
	require.Equal(t, doneItems, terminalOutput,
		"terminal response.output must deep-equal the delivered done items")
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

// Preserve existing summary indices while repairing the missing item lifecycle.
func TestParateraDeepSeekRepairKeepsExistingSummaryIndex(t *testing.T) {
	const upstream = `data: {"type":"response.reasoning_summary_text.delta","item_id":"rs_1","output_index":0,"summary_index":0,"delta":"think","model":"DeepSeek-V4.1-Flash"}

`
	payloads := ssePayloads(t, runParateraRepair(t, upstream))
	deltas := 0
	for _, payload := range payloads {
		if gjson.Get(payload, "type").String() == "response.reasoning_summary_text.delta" {
			deltas++
			require.Equal(t, int64(0), gjson.Get(payload, "summary_index").Int())
			require.Equal(t, "think", gjson.Get(payload, "delta").String())
		}
	}
	require.Equal(t, 1, deltas)
}

// Counterexample from the protocol audit: the upstream announces an empty
// message item in the middle of the reasoning phase, then keeps streaming
// reasoning deltas, and finally repeats the full reasoning in the terminal
// state. The reasoning content persisted for the client must be "AB" exactly
// once -- not the truncated "A" and not "ABB".
func TestParateraDeepSeekRepairKeepsReasoningAcrossEarlyMessageAnnouncement(t *testing.T) {
	const upstream = `data: {"type":"response.created","response":{"id":"resp_early"},"model":"DeepSeek-V4.1-Flash"}

data: {"type":"response.output_item.added","output_index":0,"item":{"id":"rs_early","type":"reasoning","status":"in_progress"},"model":"DeepSeek-V4.1-Flash"}

data: {"type":"response.reasoning_summary_text.delta","item_id":"rs_early","output_index":0,"summary_index":0,"delta":"A","model":"DeepSeek-V4.1-Flash"}

data: {"type":"response.output_item.added","output_index":0,"item":{"id":"msg_early","type":"message","role":"assistant","status":"in_progress","content":[]},"model":"DeepSeek-V4.1-Flash"}

data: {"type":"response.reasoning_summary_text.delta","item_id":"rs_early","output_index":0,"summary_index":0,"delta":"B","model":"DeepSeek-V4.1-Flash"}

data: {"type":"response.completed","response":{"id":"resp_early","status":"completed","output":[{"id":"rs_early","type":"reasoning","status":"completed","summary":[{"type":"summary_text","text":"AB"}]}],"usage":{"input_tokens":10,"output_tokens":2}}}

`
	repaired := runParateraRepair(t, upstream)
	payloads := ssePayloads(t, repaired)

	var doneTexts []string
	for _, payload := range payloads {
		if gjsonGetString(payload, "type") == "response.reasoning_summary_text.done" {
			doneTexts = append(doneTexts, gjsonGetString(payload, "text"))
		}
	}
	require.Equal(t, []string{"AB"}, doneTexts,
		"reasoning summary must be persisted once and complete")
	require.Equal(t, "AB",
		gjsonGetString(payloads[len(payloads)-1], "response.output.0.summary.0.text"),
		"terminal reasoning item must carry the same complete content")
	require.Equal(t, 1, countEventType(eventTypes(payloads), "response.reasoning_summary_text.done"),
		"reasoning completion must be emitted exactly once")
}

// The upstream can carry reasoning only in the terminal snapshot, with no
// streamed reasoning delta at all. Codex persists output_item.done, so the item
// must still be announced and closed for the client.
func TestParateraDeepSeekRepairEmitsTerminalOnlyReasoning(t *testing.T) {
	const upstream = `data: {"type":"response.created","response":{"id":"resp_term"},"model":"DeepSeek-V4.1-Flash"}

data: {"type":"response.output_text.delta","item_id":"msg_term","output_index":0,"content_index":0,"delta":"hi","model":"DeepSeek-V4.1-Flash"}

data: {"type":"response.output_text.done","item_id":"msg_term","output_index":0,"content_index":0,"text":"hi","model":"DeepSeek-V4.1-Flash"}

data: {"type":"response.completed","response":{"id":"resp_term","status":"completed","output":[{"id":"rs_term","type":"reasoning","status":"completed","summary":[{"type":"summary_text","text":"AB"}]},{"id":"msg_term","type":"message","role":"assistant","content":[{"type":"output_text","text":"hi"}]}],"usage":{"input_tokens":5,"output_tokens":3}}}

`
	repaired := runParateraRepair(t, upstream)
	payloads := ssePayloads(t, repaired)
	require.Equal(t, 1, countEventType(eventTypes(payloads), "response.reasoning_summary_text.done"))
	require.Equal(t, "AB", gjsonGetString(payloads[len(payloads)-1], "response.output.0.summary.0.text"))
}

// Two genuine reasoning items in one response must stay two canonical items.
func TestParateraDeepSeekRepairKeepsSeparateReasoningItems(t *testing.T) {
	const upstream = `data: {"type":"response.created","response":{"id":"resp_two"},"model":"DeepSeek-V4.1-Flash"}

data: {"type":"response.output_item.added","output_index":0,"item":{"id":"rs_one","type":"reasoning","status":"in_progress"},"model":"DeepSeek-V4.1-Flash"}

data: {"type":"response.reasoning_summary_text.delta","item_id":"rs_one","output_index":0,"summary_index":0,"delta":"A","model":"DeepSeek-V4.1-Flash"}

data: {"type":"response.output_item.done","output_index":0,"item":{"id":"rs_one","type":"reasoning","status":"completed","summary":[{"type":"summary_text","text":"A"}]},"model":"DeepSeek-V4.1-Flash"}

data: {"type":"response.output_item.added","output_index":1,"item":{"id":"rs_two","type":"reasoning","status":"in_progress"},"model":"DeepSeek-V4.1-Flash"}

data: {"type":"response.reasoning_summary_text.delta","item_id":"rs_two","output_index":1,"summary_index":0,"delta":"B","model":"DeepSeek-V4.1-Flash"}

data: {"type":"response.completed","response":{"id":"resp_two","status":"completed","output":[{"id":"rs_one_terminal","type":"reasoning","status":"completed","summary":[{"type":"summary_text","text":"A"}]},{"id":"rs_two_terminal","type":"reasoning","status":"completed","summary":[{"type":"summary_text","text":"B"}]}],"usage":{"input_tokens":5,"output_tokens":3}}}

`
	repaired := runParateraRepair(t, upstream)
	payloads := ssePayloads(t, repaired)
	var doneTexts []string
	for _, payload := range payloads {
		if gjsonGetString(payload, "type") == "response.reasoning_summary_text.done" {
			doneTexts = append(doneTexts, gjsonGetString(payload, "text"))
		}
	}
	require.Equal(t, []string{"A", "B"}, doneTexts)
	terminal := payloads[len(payloads)-1]
	require.Equal(t, "A", gjsonGetString(terminal, "response.output.0.summary.0.text"))
	require.Equal(t, "B", gjsonGetString(terminal, "response.output.1.summary.0.text"))
}

// The upstream is known to invent a new reasoning id per delta; all of those
// ids must resolve to one canonical item, not spawn new ones.
func TestParateraDeepSeekRepairMergesUpstreamIDChurn(t *testing.T) {
	const upstream = `data: {"type":"response.output_item.added","output_index":0,"item":{"id":"rs_first","type":"reasoning","status":"in_progress"},"model":"DeepSeek-V4.1-Flash"}

data: {"type":"response.reasoning_summary_text.delta","item_id":"rs_a","output_index":0,"summary_index":0,"delta":"A","model":"DeepSeek-V4.1-Flash"}

data: {"type":"response.reasoning_summary_text.delta","item_id":"rs_b","output_index":0,"summary_index":0,"delta":"B","model":"DeepSeek-V4.1-Flash"}

data: {"type":"response.completed","response":{"id":"resp_churn","status":"completed","output":[{"id":"rs_terminal","type":"reasoning","status":"completed","summary":[{"type":"summary_text","text":"AB"}]}],"usage":{"input_tokens":5,"output_tokens":3}}}

`
	repaired := runParateraRepair(t, upstream)
	payloads := ssePayloads(t, repaired)
	require.Equal(t, 1, countEventType(eventTypes(payloads), "response.reasoning_summary_text.done"))
	require.Equal(t, "AB", gjsonGetString(payloads[len(payloads)-1], "response.output.0.summary.0.text"))
	for _, payload := range payloads {
		if gjsonGetString(payload, "type") == "response.reasoning_summary_text.delta" {
			require.Equal(t, "rs_first", gjsonGetString(payload, "item_id"),
				"every delta must be rewritten to the canonical id")
		}
	}
}

// Reasoning and the message can share the same output_index upstream; the text
// delta must still have an owning announcement and the reasoning must survive.
func TestParateraDeepSeekRepairHandlesSharedOutputIndex(t *testing.T) {
	const upstream = `data: {"type":"response.output_item.added","output_index":0,"item":{"id":"rs_shared","type":"reasoning","status":"in_progress"},"model":"DeepSeek-V4.1-Flash"}

data: {"type":"response.reasoning_summary_text.delta","item_id":"rs_shared","output_index":0,"summary_index":0,"delta":"think","model":"DeepSeek-V4.1-Flash"}

data: {"type":"response.output_item.added","output_index":0,"item":{"id":"msg_shared","type":"message","role":"assistant","status":"in_progress","content":[]},"model":"DeepSeek-V4.1-Flash"}

data: {"type":"response.output_text.delta","item_id":"msg_shared","output_index":0,"content_index":0,"delta":"hi","model":"DeepSeek-V4.1-Flash"}

data: {"type":"response.completed","response":{"id":"resp_shared","status":"completed","output":[{"id":"rs_shared_terminal","type":"reasoning","status":"completed","summary":[{"type":"summary_text","text":"think"}]},{"id":"msg_shared","type":"message","role":"assistant","content":[{"type":"output_text","text":"hi"}]}],"usage":{"input_tokens":5,"output_tokens":3}}}

`
	repaired := runParateraRepair(t, upstream)
	payloads := ssePayloads(t, repaired)
	types := eventTypes(payloads)
	require.Equal(t, 1, countEventType(types, "response.reasoning_summary_text.done"))
	// the text delta is owned by an announcement that precedes it
	addedAt := indexOfEventType(types, "response.output_item.added")
	deltaAt := indexOfEventType(types, "response.output_text.delta")
	require.NotEqual(t, -1, addedAt)
	require.True(t, addedAt < deltaAt, "message must be announced before its text")
	require.Equal(t, "think", gjsonGetString(payloads[len(payloads)-1], "response.output.0.summary.0.text"))
}

// The upstream reports status=incomplete on a response.completed event. The
// terminal event type must match the status, and no reason may be invented.
func TestParateraDeepSeekRepairAlignsIncompleteTerminalEventType(t *testing.T) {
	const upstream = `data: {"type":"response.completed","response":{"id":"resp_budget","status":"incomplete","output":[],"usage":{"input_tokens":10,"output_tokens":300,"output_tokens_details":{"reasoning_tokens":300}}},"model":"DeepSeek-V4.1-Flash"}

`
	repaired := runParateraRepair(t, upstream)
	payloads := ssePayloads(t, repaired)
	require.Len(t, payloads, 1)
	require.Equal(t, "response.incomplete", gjsonGetString(payloads[0], "type"))
	require.Equal(t, "incomplete", gjsonGetString(payloads[0], "response.status"))
	require.False(t, gjson.GetBytes([]byte(payloads[0]), "response.incomplete_details").Exists(),
		"the upstream sent no reason, so none may be invented")
}

// Reasoning deltas that arrive after the item was closed for the client cannot
// be applied. They must be counted instead of silently disappearing.
func TestParateraDeepSeekRepairCountsLateReasoningAfterClose(t *testing.T) {
	repair := newParateraDeepSeekItemLifecycleRepair()
	repair.Repair([]byte(`{"type":"response.output_item.added","output_index":0,"item":{"id":"rs_late","type":"reasoning","status":"in_progress"}}`))
	repair.Repair([]byte(`{"type":"response.reasoning_summary_text.delta","item_id":"rs_late","output_index":0,"summary_index":0,"delta":"A"}`))
	repair.finishReasoning()
	repair.Repair([]byte(`{"type":"response.reasoning_summary_text.delta","item_id":"rs_late","output_index":0,"summary_index":0,"delta":"B"}`))
	require.Equal(t, 1, repair.lateReasoningDeltas)
	require.Equal(t, "A", repair.reasoning.text())
	require.True(t, repair.protocolFailure, "late reasoning after close must be an explicit protocol failure")
}

// A stream that stops early must not fabricate a terminal state.
func TestParateraDeepSeekRepairSurvivesTruncatedStream(t *testing.T) {
	// EOF after a complete event: the message must still be announced, and no
	// terminal state may be invented.
	const cutAfterEvent = `data: {"type":"response.created","response":{"id":"resp_cut"},"model":"DeepSeek-V4.1-Flash"}

data: {"type":"response.output_text.delta","item_id":"msg_cut","output_index":0,"content_index":0,"delta":"par","model":"DeepSeek-V4.1-Flash"}

`
	repaired := runParateraRepair(t, cutAfterEvent)
	payloads := ssePayloads(t, repaired)
	types := eventTypes(payloads)
	require.NotContains(t, types, "response.completed")
	require.NotContains(t, types, "response.incomplete")
	require.NotEqual(t, -1, indexOfEventType(types, "response.output_item.added"),
		"the message must still be announced for the text delta")

	// A stream cut in the middle of a JSON payload must not crash the repair and
	// must not fabricate anything for the unparsable tail.
	const cutMidPayload = `data: {"type":"response.created","response":{"id":"resp_cut2"},"model":"DeepSeek-V4.1-Flash"}

data: {"type":"response.output_text.delta","item_id":"msg_cut2","output_index":0,"content_index":0,"delta":"par`
	repaired = runParateraRepair(t, cutMidPayload)
	payloads = ssePayloads(t, repaired)
	types = eventTypes(payloads)
	require.NotContains(t, types, "response.completed")
	require.NotContains(t, types, "response.incomplete")
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

// A late reasoning delta must not be silently dropped and then reported as a
// successful completion: the terminal event is rewritten to response.failed.
func TestParateraDeepSeekRepairLateReasoningFailsTerminal(t *testing.T) {
	const upstream = `data: {"type":"response.output_item.added","output_index":0,"item":{"id":"rs_late","type":"reasoning","status":"in_progress"},"model":"DeepSeek-V4.1-Flash"}

data: {"type":"response.reasoning_summary_text.delta","item_id":"rs_late","output_index":0,"summary_index":0,"delta":"A","model":"DeepSeek-V4.1-Flash"}

data: {"type":"response.output_item.done","output_index":0,"item":{"id":"rs_late","type":"reasoning","status":"completed","summary":[{"type":"summary_text","text":"A"}]},"model":"DeepSeek-V4.1-Flash"}

data: {"type":"response.reasoning_summary_text.delta","item_id":"rs_late","output_index":0,"summary_index":0,"delta":"B","model":"DeepSeek-V4.1-Flash"}

data: {"type":"response.completed","response":{"id":"resp_late","status":"completed","output":[{"id":"rs_late","type":"reasoning","status":"completed","summary":[{"type":"summary_text","text":"AB"}]}],"usage":{"input_tokens":5,"output_tokens":3}},"model":"DeepSeek-V4.1-Flash"}

`
	repaired := runParateraRepair(t, upstream)
	payloads := ssePayloads(t, repaired)
	require.NotEmpty(t, payloads)
	terminal := payloads[len(payloads)-1]
	require.Equal(t, "response.failed", gjsonGetString(terminal, "type"),
		"a late reasoning delta must not be reported as a successful completion")
	require.Equal(t, "failed", gjsonGetString(terminal, "response.status"))
}

// A terminal snapshot whose reasoning differs from the already-delivered item is
// unrecoverable: it must fail explicitly, not silently keep the stale content.
func TestParateraDeepSeekRepairTerminalMismatchFails(t *testing.T) {
	const upstream = `data: {"type":"response.output_item.added","output_index":0,"item":{"id":"rs_mm","type":"reasoning","status":"in_progress"},"model":"DeepSeek-V4.1-Flash"}

data: {"type":"response.reasoning_summary_text.delta","item_id":"rs_mm","output_index":0,"summary_index":0,"delta":"A","model":"DeepSeek-V4.1-Flash"}

data: {"type":"response.output_item.done","output_index":0,"item":{"id":"rs_mm","type":"reasoning","status":"completed","summary":[{"type":"summary_text","text":"A"}]},"model":"DeepSeek-V4.1-Flash"}

data: {"type":"response.completed","response":{"id":"resp_mm","status":"completed","output":[{"id":"rs_mm_term","type":"reasoning","status":"completed","summary":[{"type":"summary_text","text":"AB"}]}],"usage":{"input_tokens":5,"output_tokens":3}},"model":"DeepSeek-V4.1-Flash"}

`
	repaired := runParateraRepair(t, upstream)
	payloads := ssePayloads(t, repaired)
	terminal := payloads[len(payloads)-1]
	require.Equal(t, "response.failed", gjsonGetString(terminal, "type"))
	require.Equal(t, "failed", gjsonGetString(terminal, "response.status"))
}

// The stream-repair gate must use the actual outbound body model (no
// re-mapping), while the entry compatibility gate resolves the client alias.
func TestParateraDeepSeekRepairGateUsesOutboundModel(t *testing.T) {
	const base = "https://llmapi.paratera.com/v1"
	mk := func(mapping map[string]any) *Account {
		creds := map[string]any{"api_key": "sk-test", "base_url": base}
		if mapping != nil {
			creds["model_mapping"] = mapping
		}
		return &Account{Platform: PlatformOpenAI, Type: AccountTypeAPIKey, Credentials: creds}
	}

	// Stream repair gates on the outbound model directly.
	plain := mk(nil)
	require.True(t, shouldRepairParateraDeepSeekStream(plain, "DeepSeek-V4.1-Flash"))
	require.False(t, shouldRepairParateraDeepSeekStream(plain, "deepseek-flash-alias"),
		"the stream gate must not re-map: an alias is not the verified outbound model")

	// The entry compatibility gate resolves the client alias once.
	aliased := mk(map[string]any{"deepseek-flash-alias": "DeepSeek-V4.1-Flash"})
	body := []byte(`{"model":"deepseek-flash-alias"}`)
	require.True(t, shouldUseParateraDeepSeekResponsesCompatibility(aliased, body),
		"the entry gate must resolve an alias that maps to the verified model")

	other := mk(map[string]any{"deepseek-flash-alias": "some-other-model"})
	require.False(t, shouldUseParateraDeepSeekResponsesCompatibility(other, body),
		"an alias mapping away from the verified model must not inherit the repair")
}

// The passthrough path does not rewrite body.model, so the repair decision must
// match the model actually sent upstream.
func TestParateraDeepSeekPassthroughGateMatchesOutboundBodyModel(t *testing.T) {
	run := func(model string) string {
		gin.SetMode(gin.TestMode)
		body := []byte(`{"model":"` + model + `","stream":true,"input":"hi"}`)
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
		_, err := svc.forwardOpenAIPassthrough(context.Background(), c, parateraStreamingTestAccount(), body, body, model, false, nil, true, time.Now())
		require.NoError(t, err)
		return recorder.Body.String()
	}

	verified := run("DeepSeek-V4.1-Flash")
	types := eventTypes(ssePayloads(t, verified))
	deltaIndex := indexOfEventType(types, "response.output_text.delta")
	require.Greater(t, deltaIndex, 1)
	require.Equal(t, "response.output_item.added", types[deltaIndex-2],
		"the verified outbound model must receive the message-announcement repair")

	aliased := run("deepseek-flash-alias")
	typesAlias := eventTypes(ssePayloads(t, aliased))
	deltaIndexAlias := indexOfEventType(typesAlias, "response.output_text.delta")
	require.Greater(t, deltaIndexAlias, 0)
	require.NotEqual(t, "response.output_item.added", typesAlias[deltaIndexAlias-1],
		"a non-verified outbound model must not be rewritten")
}

// A late reasoning delta must not be silently dropped and then reported as a
// successful completion: the terminal event is rewritten to response.failed.

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

func TestParateraPassthroughActuallySendsReasoningContent(t *testing.T) {
	body := []byte(`{"model":"DeepSeek-V4.1-Flash","stream":true,"store":false,"input":[{"type":"reasoning","id":"rs_saved","summary":[{"type":"summary_text","text":"Keep NEBULA_72639"}]},{"role":"user","content":"What was the marker?"}]}`)
	recorder := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(recorder)
	c.Request = httptest.NewRequest(http.MethodPost, "/v1/responses", bytes.NewReader(body))
	upstream := &httpUpstreamRecorder{resp: &http.Response{
		StatusCode: http.StatusOK, Header: http.Header{"Content-Type": []string{"text/event-stream"}},
		Body: io.NopCloser(strings.NewReader(brokenParateraStream)),
	}}
	svc := openAIClientToolsTestService(upstream)
	_, err := svc.forwardOpenAIPassthrough(context.Background(), c, parateraStreamingTestAccount(), body, body, "DeepSeek-V4.1-Flash", false, nil, true, time.Now())
	require.NoError(t, err)
	require.Equal(t, "reasoning", gjson.GetBytes(upstream.lastBody, "input.0.type").String())
	require.Equal(t, "Keep NEBULA_72639", gjson.GetBytes(upstream.lastBody, "input.0.content.0.text").String())
}

// When a reasoning item carries both representations and they disagree, the
// content the client just replayed wins; a stale non-empty content must not be
// forwarded silently.
func TestParateraReasoningReplayPrefersClientReplayOnMismatch(t *testing.T) {
	body := []byte(`{"model":"DeepSeek-V4.1-Flash","store":false,"input":[` +
		`{"type":"reasoning","id":"rs_x","summary":[{"type":"summary_text","text":"NEW"}],` +
		`"content":[{"type":"output_text","text":"OLD"}]}]}`)
	normalized, changed, err := normalizeParateraDeepSeekReasoningReplay(body)
	require.NoError(t, err)
	require.True(t, changed)
	require.Equal(t, "NEW", gjson.GetBytes(normalized, "input.0.content.0.text").String())
	require.Equal(t, "output_text", gjson.GetBytes(normalized, "input.0.content.0.type").String())

	// Agreement is left untouched.
	same := []byte(`{"model":"DeepSeek-V4.1-Flash","store":false,"input":[` +
		`{"type":"reasoning","id":"rs_y","summary":[{"type":"summary_text","text":"SAME"}],` +
		`"content":[{"type":"output_text","text":"SAME"}]}]}`)
	unchanged, changed, err := normalizeParateraDeepSeekReasoningReplay(same)
	require.NoError(t, err)
	require.False(t, changed)
	require.JSONEq(t, string(same), string(unchanged))
}

func TestParateraReasoningReplaySurvivesPassthroughNormalization(t *testing.T) {
	body := []byte(`{"model":"DeepSeek-V4.1-Flash","store":false,"input":[{"type":"reasoning","id":"rs_saved","summary":[{"type":"summary_text","text":"marker=NEBULA_72639"}]},{"type":"function_call","name":"lookup","call_id":"call_a","arguments":"{}"},{"type":"function_call_output","call_id":"call_a","output":"ok"}]}`)
	converted, changed, err := normalizeParateraDeepSeekReasoningReplay(body)
	require.NoError(t, err)
	require.True(t, changed)
	normalized, _, err := normalizeOpenAIResponsesWebSocketCompatibilityBody(converted, parateraStreamingTestAccount(), false)
	require.NoError(t, err)
	require.Equal(t, 3, len(gjson.GetBytes(normalized, "input").Array()))
	require.Equal(t, "marker=NEBULA_72639", gjson.GetBytes(normalized, "input.0.content.0.text").String())
	require.Equal(t, "output_text", gjson.GetBytes(normalized, "input.0.content.0.type").String())
	twice, changed, err := normalizeParateraDeepSeekReasoningReplay(normalized)
	require.NoError(t, err)
	require.False(t, changed)
	require.Equal(t, normalized, twice)
	account := parateraStreamingTestAccount()
	account.Credentials["base_url"] = "https://api.openai.com/v1"
	other, _, err := normalizeOpenAIResponsesWebSocketCompatibilityBody(converted, account, false)
	require.NoError(t, err)
	require.Equal(t, "function_call", gjson.GetBytes(other, "input.0.type").String())
}

func TestParateraReasoningMissingLifecyclePersistsBeforeTool(t *testing.T) {
	stream := `data: {"type":"response.reasoning_summary_text.delta","item_id":"rs_random1","output_index":0,"delta":"Remember "}

data: {"type":"response.reasoning_summary_text.delta","item_id":"rs_random2","output_index":0,"delta":"NEBULA_72639"}

data: {"type":"response.output_item.added","output_index":1,"item":{"type":"function_call","id":"fc_a","call_id":"call_a","name":"lookup","arguments":""}}

data: {"type":"response.output_item.done","output_index":1,"item":{"type":"function_call","id":"fc_a","call_id":"call_a","name":"lookup","arguments":"{}"}}

data: {"type":"response.completed","response":{"id":"resp_a","output":[{"type":"reasoning","id":"rs_terminal","content":[{"type":"output_text","text":"Remember NEBULA_72639"}]}]}}

`
	payloads := ssePayloads(t, runParateraRepair(t, stream))
	var id string
	var reasoningDone int
	for _, payload := range payloads {
		e := gjson.Parse(payload)
		switch e.Get("type").String() {
		case "response.output_item.added":
			if e.Get("item.type").String() == "reasoning" {
				id = e.Get("item.id").String()
				require.True(t, e.Get("item.summary").IsArray(), "Codex requires summary even on added")
			} else {
				require.Equal(t, 1, reasoningDone, "persist before executing tool")
			}
		case "response.reasoning_summary_text.delta":
			require.NotEmpty(t, id)
			require.Equal(t, id, e.Get("item_id").String())
			require.True(t, e.Get("summary_index").Exists())
		case "response.output_item.done":
			if e.Get("item.type").String() == "reasoning" {
				reasoningDone++
				require.Equal(t, "Remember NEBULA_72639", e.Get("item.summary.0.text").String())
				require.Equal(t, id, e.Get("item.id").String())
			}
		case "response.completed":
			require.Equal(t, id, e.Get("response.output.0.id").String())
			require.Equal(t, "Remember NEBULA_72639", e.Get("response.output.0.summary.0.text").String())
		}
	}
	require.Equal(t, 1, reasoningDone)
}

func TestParateraTerminalOnlyReasoningRoundTrips(t *testing.T) {
	stream := "data: " + `{"type":"response.completed","response":{"id":"resp_a","output":[{"type":"reasoning","id":"rs_terminal","content":[{"type":"output_text","text":"unique-marker"}]}]}}` + "\n\n"
	payloads := ssePayloads(t, runParateraRepair(t, stream))
	types := eventTypes(payloads)
	require.Equal(t, 1, countEventType(types, "response.output_item.done"))
	require.Equal(t, "response.completed", types[len(types)-1])
	for _, payload := range payloads {
		if gjson.Get(payload, "type").String() != "response.output_item.done" {
			continue
		}
		body := []byte(`{"input":[` + gjson.Get(payload, "item").Raw + `]}`)
		upstream, _, err := normalizeParateraDeepSeekReasoningReplay(body)
		require.NoError(t, err)
		require.Equal(t, "unique-marker", gjson.GetBytes(upstream, "input.0.content.0.text").String())
	}
}

func TestParateraReasoningLateDoneDoesNotDuplicateHistory(t *testing.T) {
	stream := strings.Replace(brokenParateraStream,
		`"id":"rs_live","type":"reasoning","status":"in_progress"`,
		`"id":"rs_live","type":"reasoning","status":"in_progress","summary":[]`, 1)
	// Existing valid done plus terminal-only alias must still persist once.
	stream = strings.Replace(stream, `"output":[{"id":"msg_live"`,
		`"output":[{"id":"rs_alias","type":"reasoning","content":[{"type":"output_text","text":"thinking"}]},{"id":"msg_live"`, 1)
	n := 0
	for _, payload := range ssePayloads(t, runParateraRepair(t, stream)) {
		if gjson.Get(payload, "type").String() == "response.output_item.done" && gjson.Get(payload, "item.type").String() == "reasoning" {
			n++
			require.Equal(t, "thinking", gjson.Get(payload, "item.summary.0.text").String())
		}
	}
	require.Equal(t, 1, n)
}

func TestParateraMessageAnnouncedBeforeReasoningKeepsTextOwner(t *testing.T) {
	// Production announces an empty message, emits reasoning, then emits its
	// text. Opening reasoning must not leave that text without an active item.
	stream := `data: {"type":"response.output_item.added","output_index":0,"item":{"type":"message","id":"msg_early","role":"assistant","content":[]}}

data: {"type":"response.content_part.added","output_index":0,"item_id":"msg_early","content_index":0,"part":{"type":"output_text","text":""}}

data: {"type":"response.reasoning_summary_text.delta","output_index":0,"item_id":"rs_random","delta":"I should call lookup."}

data: {"type":"response.output_text.delta","output_index":0,"item_id":"msg_early","content_index":0,"delta":"Calling lookup."}

data: {"type":"response.output_item.done","output_index":0,"item":{"type":"message","id":"msg_early","role":"assistant","content":[{"type":"output_text","text":"Calling lookup."}]}}

data: {"type":"response.completed","response":{"id":"resp_a","output":[]}}

`
	active := ""
	messageAnnouncements := 0
	textDeltas := 0
	for _, payload := range ssePayloads(t, runParateraRepair(t, stream)) {
		e := gjson.Parse(payload)
		switch e.Get("type").String() {
		case "response.output_item.added":
			active = e.Get("item.id").String()
			if e.Get("item.type").String() == "message" {
				messageAnnouncements++
			}
		case "response.output_item.done":
			active = ""
		case "response.output_text.delta", "response.reasoning_summary_text.delta":
			require.NotEmpty(t, active)
			require.Equal(t, e.Get("item_id").String(), active)
			if e.Get("type").String() == "response.output_text.delta" {
				textDeltas++
			}
		}
	}
	require.Equal(t, 1, messageAnnouncements)
	require.Equal(t, 1, textDeltas)
}

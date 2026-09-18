package apicompat

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"
)

// commandCodeStreamChunks loads a captured stream from api.commandcode.ai, a
// provider that only speaks Chat Completions and reports its thinking in the
// non-standard `reasoning` field (plus `reasoning_details`) instead of
// `reasoning_content`.
func commandCodeStreamChunks(t *testing.T) []string {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("testdata", "commandcode", "tool_call_reasoning_stream.sse"))
	require.NoError(t, err)
	var chunks []string
	for _, line := range strings.Split(string(raw), "\n") {
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		body := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if body == "" || body == "[DONE]" {
			continue
		}
		chunks = append(chunks, body)
	}
	require.NotEmpty(t, chunks)
	return chunks
}

// The `reasoning` alias must reach the client as reasoning events, and every
// such delta must reference an item the bridge has already opened: Codex drops
// reasoning deltas whose item is not open.
func TestCommandCodeStream_ReasoningAliasOpensItemBeforeDelta(t *testing.T) {
	events := collectStreamEvents(t, commandCodeStreamChunks(t))

	open := map[int]string{} // output_index -> item type
	reasoningText := ""
	var toolCalls []string
	for _, e := range events {
		switch e.Type {
		case "response.output_item.added":
			require.NotNil(t, e.Item)
			open[e.OutputIndex] = e.Item.Type
		case "response.reasoning_summary_text.delta":
			require.Equalf(t, "reasoning", open[e.OutputIndex],
				"reasoning delta before its item was opened")
			reasoningText += e.Delta
		case "response.output_item.done":
			if e.Item != nil && e.Item.Type == "function_call" {
				toolCalls = append(toolCalls, e.Item.Name)
			}
		}
	}
	require.Contains(t, reasoningText, "lookup",
		"the upstream reasoning must survive the alias mapping")
	require.Contains(t, toolCalls, "lookup",
		"the tool call must be bridged as a function_call item")
}

// A non-streaming response from the same provider must expose reasoning the
// same way, so a client that only sees the terminal payload still gets it.
func TestCommandCodeNonStreaming_ReasoningAliasBecomesReasoningItem(t *testing.T) {
	var response ChatCompletionsResponse
	require.NoError(t, json.Unmarshal([]byte(`{
		"id":"gen_x","object":"chat.completion","model":"deepseek/deepseek-v4.1-flash",
		"choices":[{"index":0,"finish_reason":"tool_calls","message":{
			"role":"assistant","content":null,
			"reasoning":"I should call lookup with key alpha.",
			"reasoning_details":[{"type":"reasoning.text","text":"I should call lookup with key alpha.","format":"unknown","index":0}],
			"tool_calls":[{"id":"call_1","type":"function","function":{"name":"lookup","arguments":"{\"key\":\"alpha\"}"}}]}}],
		"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}`), &response))

	out := ChatCompletionsResponseToResponses(&response, "deepseek/deepseek-v4.1-flash", nil, nil, false, nil)
	require.NotEmpty(t, out.Output)
	require.Equal(t, "reasoning", out.Output[0].Type)
	require.Contains(t, out.Output[0].Summary[0].Text, "lookup")
}

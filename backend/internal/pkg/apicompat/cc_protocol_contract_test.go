package apicompat

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/require"
)

// terminalEvent returns the last terminal event of a stream, failing if none was
// emitted.
func terminalEvent(t *testing.T, events []ResponsesStreamEvent) ResponsesStreamEvent {
	t.Helper()
	for i := len(events) - 1; i >= 0; i-- {
		switch events[i].Type {
		case "response.completed", "response.incomplete", "response.failed":
			return events[i]
		}
	}
	t.Fatalf("no terminal event in %d events", len(events))
	return ResponsesStreamEvent{}
}

// reasoningOnlyStream drives the Chat Completions bridge with a reasoning-only
// completion and returns the collected events.
func reasoningOnlyStream(t *testing.T, finishReason string) []ResponsesStreamEvent {
	t.Helper()
	return collectStreamEvents(t, []string{
		`{"choices":[{"index":0,"delta":{"role":"assistant","content":null,"reasoning_content":""}}]}`,
		`{"choices":[{"index":0,"delta":{"reasoning_content":"step by step"}}]}`,
		`{"choices":[{"index":0,"delta":{"content":""},"finish_reason":"` + finishReason + `"}],"usage":{"prompt_tokens":1,"completion_tokens":2,"total_tokens":3}}`,
	})
}

// The terminal event type must follow the response status: budget exhaustion is
// a response.incomplete, not a response.completed that merely carries
// status=incomplete. Clients dispatch on the event type.
func TestCCContract_ReasoningOnlyLengthIsIncompleteTerminal(t *testing.T) {
	events := reasoningOnlyStream(t, "length")

	terminal := terminalEvent(t, events)
	require.Equal(t, "response.incomplete", terminal.Type)
	require.NotNil(t, terminal.Response)
	require.Equal(t, "incomplete", terminal.Response.Status)
	require.NotNil(t, terminal.Response.IncompleteDetails)
	require.Equal(t, "max_output_tokens", terminal.Response.IncompleteDetails.Reason)
	require.NotEmpty(t, terminal.Response.Output)
	require.Equal(t, "reasoning", terminal.Response.Output[0].Type)
	require.Equal(t, "step by step", terminal.Response.Output[0].Summary[0].Text)

	for _, e := range events {
		require.NotEqual(t, "response.output_text.delta", e.Type,
			"reasoning-only turn must not fabricate assistant text")
	}
	for _, out := range terminal.Response.Output {
		if out.Type == "message" {
			require.Equal(t, "", out.Content[0].Text,
				"reasoning must not be copied into the assistant body")
		}
	}
}

// A reasoning-only turn that finished normally is a completed response; the
// reasoning item is kept and no answer text is invented.
func TestCCContract_ReasoningOnlyStopIsCompletedTerminal(t *testing.T) {
	events := reasoningOnlyStream(t, "stop")

	terminal := terminalEvent(t, events)
	require.Equal(t, "response.completed", terminal.Type)
	require.NotNil(t, terminal.Response)
	require.Equal(t, "completed", terminal.Response.Status)
	require.Nil(t, terminal.Response.IncompleteDetails)
	require.Equal(t, "reasoning", terminal.Response.Output[0].Type)
	require.Equal(t, "step by step", terminal.Response.Output[0].Summary[0].Text)
	for _, e := range events {
		require.NotEqual(t, "response.output_text.delta", e.Type,
			"reasoning-only turn must not fabricate assistant text")
	}
}

// Ordinary stop and tool_calls finishes stay completed.
func TestCCContract_StopAndToolCallsStayCompleted(t *testing.T) {
	t.Run("stop", func(t *testing.T) {
		events := collectStreamEvents(t, []string{
			`{"choices":[{"index":0,"delta":{"content":"final answer"}}]}`,
			`{"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`,
		})
		terminal := terminalEvent(t, events)
		require.Equal(t, "response.completed", terminal.Type)
		require.Equal(t, "completed", terminal.Response.Status)
		require.Nil(t, terminal.Response.IncompleteDetails)
	})

	t.Run("tool_calls", func(t *testing.T) {
		events := collectStreamEvents(t, []string{
			`{"choices":[{"index":0,"delta":{"reasoning_content":"plan"}}]}`,
			`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_a","type":"function","function":{"name":"exec","arguments":"{}"}}]}}]}`,
			`{"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}`,
		})
		terminal := terminalEvent(t, events)
		require.Equal(t, "response.completed", terminal.Type)
		require.Equal(t, "completed", terminal.Response.Status)
		require.Nil(t, terminal.Response.IncompleteDetails)
	})
}

// Non-streaming parity: a reasoning-only completion keeps the reasoning item and
// never copies it into the assistant message body, and length maps to
// status=incomplete.
func TestCCContract_NonStreamingReasoningOnlyAndLength(t *testing.T) {
	makeResp := func(finishReason string) *ChatCompletionsResponse {
		content := json.RawMessage(`""`)
		return &ChatCompletionsResponse{
			ID:    "chatcmpl_contract",
			Model: "deepseek-reasoner",
			Choices: []ChatChoice{{
				Index: 0,
				Message: ChatMessage{
					Role:             "assistant",
					Content:          content,
					ReasoningContent: "reasoning only",
				},
				FinishReason: finishReason,
			}},
		}
	}

	t.Run("length", func(t *testing.T) {
		out := ChatCompletionsResponseToResponses(makeResp("length"), "deepseek-reasoner", nil, nil, false, nil)
		require.Equal(t, "incomplete", out.Status)
		require.NotNil(t, out.IncompleteDetails)
		require.Equal(t, "max_output_tokens", out.IncompleteDetails.Reason)
		require.Equal(t, "reasoning", out.Output[0].Type)
		require.Equal(t, "reasoning only", out.Output[0].Summary[0].Text)
		for _, o := range out.Output {
			if o.Type == "message" {
				require.Equal(t, "", o.Content[0].Text)
			}
		}
	})

	t.Run("stop", func(t *testing.T) {
		out := ChatCompletionsResponseToResponses(makeResp("stop"), "deepseek-reasoner", nil, nil, false, nil)
		require.Equal(t, "completed", out.Status)
		require.Nil(t, out.IncompleteDetails)
		require.Equal(t, "reasoning", out.Output[0].Type)
		for _, o := range out.Output {
			if o.Type == "message" {
				require.Equal(t, "", o.Content[0].Text)
			}
		}
	})
}

package apicompat

// 本文件锁定"晚到 reasoning / 语义分阶段 / 工具完成边界"的 CC 合同：
//   - reasoning A -> message X -> reasoning B -> message Y 必须是 4 个独立 item，
//     added/delta/done 顺序闭合，done 顺序与终态 response.output 一致；
//   - 工具分片只在 finish_reason=tool_calls 时交付，稀疏 index 按首次出现序；
//   - 无 finish evidence / 坏协议不得伪装成功。

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/require"
)

// runCCStream 驱动状态机并在末尾 finalize，返回全部事件。
func runCCStream(t *testing.T, state *ChatCompletionsToResponsesStreamState, payloads ...string) []ResponsesStreamEvent {
	t.Helper()
	var events []ResponsesStreamEvent
	for _, p := range payloads {
		var chunk ChatCompletionsChunk
		require.NoError(t, json.Unmarshal([]byte(p), &chunk))
		events = append(events, ChatCompletionsChunkToResponsesEvents(&chunk, state)...)
	}
	events = append(events, FinalizeChatCompletionsResponsesStream(state)...)
	return events
}

// runCCStreamOK drives a stream that is expected to deliver a normal terminal
// (or incomplete) response and asserts the full delivered lifecycle.
func runCCStreamOK(t *testing.T, state *ChatCompletionsToResponsesStreamState, payloads ...string) []ResponsesStreamEvent {
	t.Helper()
	events := runCCStream(t, state, payloads...)
	assertDeliveredLifecycle(t, events)
	return events
}

// itemDoneOrder 返回 output_item.done 的 (type,id) 顺序，并校验 done 后不再有同 item 的 delta。
func itemDoneOrder(t *testing.T, events []ResponsesStreamEvent) []string {
	t.Helper()
	var order []string
	done := map[string]bool{}
	for _, e := range events {
		id := e.ItemID
		if e.Item != nil {
			id = e.Item.ID
		}
		switch e.Type {
		case "response.output_item.done":
			require.NotNil(t, e.Item)
			order = append(order, e.Item.Type+":"+e.Item.ID)
			done[e.Item.ID] = true
		case "response.reasoning_summary_text.delta", "response.output_text.delta",
			"response.function_call_arguments.delta":
			if id != "" {
				require.Falsef(t, done[id], "delta after output_item.done for item %s", id)
			}
		}
	}
	return order
}

func terminalOf(t *testing.T, events []ResponsesStreamEvent) *ResponsesStreamEvent {
	t.Helper()
	var terminal *ResponsesStreamEvent
	for i := range events {
		switch events[i].Type {
		case "response.completed", "response.incomplete":
			evt := events[i]
			terminal = &evt
		}
	}
	return terminal
}

func TestCCContract_ReasoningMessageReasoningMessageAreOrderedItems(t *testing.T) {
	state := NewChatCompletionsToResponsesStreamState("deepseek-v4-pro")
	events := runCCStreamOK(t, state,
		`{"choices":[{"index":0,"delta":{"reasoning_content":"A"}}]}`,
		`{"choices":[{"index":0,"delta":{"content":"X"}}]}`,
		`{"choices":[{"index":0,"delta":{"reasoning_content":"B"}}]}`,
		`{"choices":[{"index":0,"delta":{"content":"Y"}}]}`,
		`{"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`,
	)

	order := itemDoneOrder(t, events)
	require.Equal(t, []string{"reasoning", "message", "reasoning", "message"}, stripIDs(order),
		"done order must be reasoning A, message X, reasoning B, message Y")

	terminal := terminalOf(t, events)
	require.NotNil(t, terminal)
	require.Equal(t, "completed", terminal.Response.Status)
	require.Len(t, terminal.Response.Output, 4)
	require.Equal(t, "reasoning", terminal.Response.Output[0].Type)
	require.Equal(t, "A", terminal.Response.Output[0].Summary[0].Text)
	require.Equal(t, "message", terminal.Response.Output[1].Type)
	require.Equal(t, "X", terminal.Response.Output[1].Content[0].Text)
	require.Equal(t, "reasoning", terminal.Response.Output[2].Type)
	require.Equal(t, "B", terminal.Response.Output[2].Summary[0].Text)
	require.Equal(t, "message", terminal.Response.Output[3].Type)
	require.Equal(t, "Y", terminal.Response.Output[3].Content[0].Text)

	// done ids must match the terminal output ids exactly (no regeneration).
	var doneIDs []string
	for _, e := range events {
		if e.Type == "response.output_item.done" && e.Item != nil {
			doneIDs = append(doneIDs, e.Item.ID)
		}
	}
	var outIDs []string
	for _, o := range terminal.Response.Output {
		outIDs = append(outIDs, o.ID)
	}
	require.Equal(t, doneIDs, outIDs, "terminal output must reuse delivered ids in done order")
}

func stripIDs(order []string) []string {
	out := make([]string, 0, len(order))
	for _, o := range order {
		if i := indexByte(o, ':'); i >= 0 {
			out = append(out, o[:i])
		} else {
			out = append(out, o)
		}
	}
	return out
}

func indexByte(s string, b byte) int {
	for i := 0; i < len(s); i++ {
		if s[i] == b {
			return i
		}
	}
	return -1
}

func TestCCContract_ReasoningAroundToolCallKeepsOrder(t *testing.T) {
	state := NewChatCompletionsToResponsesStreamState("deepseek-v4-pro")
	events := runCCStreamOK(t, state,
		`{"choices":[{"index":0,"delta":{"reasoning_content":"before tool"}}]}`,
		`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_a","type":"function","function":{"name":"exec","arguments":"{}"}}]}}]}`,
		`{"choices":[{"index":0,"delta":{"reasoning_content":"after tool"}}]}`,
		`{"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}`,
	)

	order := stripIDs(itemDoneOrder(t, events))
	require.Equal(t, []string{"reasoning", "function_call", "reasoning"}, order,
		"reasoning before and after a tool call must stay on opposite sides of it")

	terminal := terminalOf(t, events)
	require.NotNil(t, terminal)
	require.Len(t, terminal.Response.Output, 3)
	require.Equal(t, "reasoning", terminal.Response.Output[0].Type)
	require.Equal(t, "function_call", terminal.Response.Output[1].Type)
	require.Equal(t, "reasoning", terminal.Response.Output[2].Type)
}

func TestCCContract_SparseToolIndicesDeliverInFirstSeenOrder(t *testing.T) {
	state := NewChatCompletionsToResponsesStreamState("deepseek-v4-pro")
	events := runCCStreamOK(t, state,
		// sparse indices 2 then 5, fragments interleaved
		`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":2,"id":"call_2","type":"function","function":{"name":"exec","arguments":"{\"a\":"}}]}}]}`,
		`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":5,"id":"call_5","type":"function","function":{"name":"exec","arguments":"{\"b\":"}}]}}]}`,
		`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":2,"function":{"arguments":"1}"}}]}}]}`,
		`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":5,"function":{"arguments":"2}"}}]}}]}`,
		`{"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}`,
	)

	var callIDs []string
	for _, e := range events {
		if e.Type == "response.output_item.done" && e.Item != nil && e.Item.Type == "function_call" {
			callIDs = append(callIDs, e.Item.CallID)
		}
	}
	require.Equal(t, []string{"call_2", "call_5"}, callIDs)

	terminal := terminalOf(t, events)
	require.Len(t, terminal.Response.Output, 2)
	require.Equal(t, `{"a":1}`, terminal.Response.Output[0].Arguments)
	require.Equal(t, `{"b":2}`, terminal.Response.Output[1].Arguments)
}

func TestCCContract_ToolWithoutFinishEvidenceIsNotDelivered(t *testing.T) {
	state := NewChatCompletionsToResponsesStreamState("deepseek-v4-pro")
	events := runCCStream(t, state,
		`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_x","type":"function","function":{"name":"exec","arguments":"{}"}}]}}]}`,
		// no finish_reason at all -> truncated stream
	)
	require.Error(t, state.ProtocolError, "a stream that ends without finish evidence must be a protocol error")
	for _, e := range events {
		require.NotEqual(t, "response.function_call_arguments.done", e.Type)
		require.NotEqual(t, "response.output_item.done", e.Type)
		require.NotEqual(t, "response.completed", e.Type)
		require.NotEqual(t, "response.incomplete", e.Type)
	}
}

func TestCCContract_StopWithPendingToolIsProtocolError(t *testing.T) {
	state := NewChatCompletionsToResponsesStreamState("deepseek-v4-pro")
	events := runCCStream(t, state,
		`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_x","type":"function","function":{"name":"exec","arguments":"{}"}}]}}]}`,
		`{"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`,
	)
	require.Error(t, state.ProtocolError)
	for _, e := range events {
		require.NotEqual(t, "response.function_call_arguments.done", e.Type)
		require.NotEqual(t, "response.completed", e.Type)
	}
}

func TestCCContract_ReasoningOnlyLengthIsIncompleteWithoutTools(t *testing.T) {
	state := NewChatCompletionsToResponsesStreamState("deepseek-v4-pro")
	events := runCCStreamOK(t, state,
		`{"choices":[{"index":0,"delta":{"reasoning_content":"think"}}]}`,
		`{"choices":[{"index":0,"delta":{},"finish_reason":"length"}]}`,
	)
	terminal := terminalOf(t, events)
	require.NotNil(t, terminal)
	require.Equal(t, "response.incomplete", terminal.Type)
	require.Equal(t, "incomplete", terminal.Response.Status)
	require.NotNil(t, terminal.Response.IncompleteDetails)
}

func TestCCContract_UsageReasoningTokensMapped(t *testing.T) {
	usage := &ChatUsage{
		PromptTokens:     10,
		CompletionTokens: 20,
		TotalTokens:      30,
		CompletionTokensDetails: &ChatTokenDetails{
			ReasoningTokens: 7,
		},
	}
	out := ChatUsageToResponsesUsage(usage)
	require.NotNil(t, out.OutputTokensDetails)
	require.Equal(t, 7, out.OutputTokensDetails.ReasoningTokens)
	require.Equal(t, 20, out.OutputTokens)
}

func TestCCContract_ReasoningAliasNotDoubleConcatenated(t *testing.T) {
	state := NewChatCompletionsToResponsesStreamState("deepseek-v4-pro")
	// reasoning_content takes precedence; the same chunk's reasoning alias must
	// not be appended on top of it.
	events := runCCStreamOK(t, state,
		`{"choices":[{"index":0,"delta":{"reasoning_content":"primary","reasoning":"alias"}}]}`,
		`{"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`,
	)
	terminal := terminalOf(t, events)
	require.NotNil(t, terminal)
	require.Len(t, terminal.Response.Output, 2)
	require.Equal(t, "reasoning", terminal.Response.Output[0].Type)
	require.Equal(t, "primary", terminal.Response.Output[0].Summary[0].Text)
}

// outputIndices 返回 output_item.done 的 output_index 序列。
func outputIndices(events []ResponsesStreamEvent) []int {
	var out []int
	for _, e := range events {
		if e.Type == "response.output_item.done" {
			out = append(out, e.OutputIndex)
		}
	}
	return out
}

func TestCCContract_ToolReasoningToolIndicesAreContiguous(t *testing.T) {
	state := NewChatCompletionsToResponsesStreamState("deepseek-v4-pro")
	events := runCCStreamOK(t, state,
		`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_a","type":"function","function":{"name":"exec","arguments":"{}"}}]},"finish_reason":null}]}`,
		`{"choices":[{"index":0,"delta":{"reasoning_content":"mid"},"finish_reason":null}]}`,
		`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"id":"call_b","type":"function","function":{"name":"exec","arguments":"{}"}}]},"finish_reason":null}]}`,
		`{"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}`,
	)
	require.Equal(t, []int{0, 1, 2}, outputIndices(events),
		"output_index must be allocated in delivery order, not at first-fragment time")
	order := stripIDs(itemDoneOrder(t, events))
	require.Equal(t, []string{"function_call", "reasoning", "function_call"}, order)
}

func TestCCContract_LengthDropsToolsWithoutIndexGaps(t *testing.T) {
	state := NewChatCompletionsToResponsesStreamState("deepseek-v4-pro")
	events := runCCStreamOK(t, state,
		`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_a","type":"function","function":{"name":"exec","arguments":"{}"}}]},"finish_reason":null}]}`,
		`{"choices":[{"index":0,"delta":{"reasoning_content":"mid"},"finish_reason":null}]}`,
		`{"choices":[{"index":0,"delta":{},"finish_reason":"length"}]}`,
	)
	require.Equal(t, []int{0, 1}, outputIndices(events),
		"dropped tools must not leave gaps in delivered output_index (reasoning + empty message)")
	order := stripIDs(itemDoneOrder(t, events))
	require.Equal(t, []string{"reasoning", "message"}, order)
	for _, e := range events {
		if e.Item != nil {
			require.NotEqual(t, "function_call", e.Item.Type, "truncated tool must not be delivered")
		}
	}
	terminal := terminalOf(t, events)
	require.Equal(t, "response.incomplete", terminal.Type)
}

func TestCCContract_BoundedBufferRejectsRunawayDeferredTail(t *testing.T) {
	state := NewChatCompletionsToResponsesStreamState("deepseek-v4-pro")
	// A tool fragment switches to deferral, then many reasoning/message phases
	// alternate; exceeding the phase cap must become an explicit protocol error.
	payloads := []string{
		`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_a","type":"function","function":{"name":"exec","arguments":"{}"}}]},"finish_reason":null}]}`,
	}
	for i := 0; i < maxDeferredPhases+5; i++ {
		payloads = append(payloads, `{"choices":[{"index":0,"delta":{"reasoning_content":"r"},"finish_reason":null}]}`)
		payloads = append(payloads, `{"choices":[{"index":0,"delta":{"content":"m"},"finish_reason":null}]}`)
	}
	payloads = append(payloads, `{"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}`)
	runCCStream(t, state, payloads...)
	require.Error(t, state.ProtocolError, "runaway deferred tail must be rejected, not buffered without bound")
}

// assertDeliveredLifecycle 校验每个已交付 item 的完整生命周期与内容，而不只是 id/type：
//   - output_index 连续 0..n-1，且与 done 顺序一致；
//   - 每个 item 先 added 后有 delta，所有 delta 拼出的内容等于 done 内容；
//   - done 之后不再有该 item 的 delta；
//   - 终态 response.output 的 id/顺序/内容与 done 完全一致。
func assertDeliveredLifecycle(t *testing.T, events []ResponsesStreamEvent) {
	t.Helper()
	added := map[string]bool{}
	done := map[string]bool{}
	deltaText := map[string]string{}
	doneText := map[string]string{}
	doneType := map[string]string{}
	var doneOrder []string
	var indexOrder []int
	nextIndex := 0

	for _, e := range events {
		switch e.Type {
		case "response.output_item.added":
			require.NotNil(t, e.Item)
			require.Equal(t, nextIndex, e.OutputIndex, "output_index must be contiguous and in delivery order")
			nextIndex++
			added[e.Item.ID] = true
			require.False(t, done[e.Item.ID], "added after done")
		case "response.reasoning_summary_text.delta", "response.output_text.delta":
			id := e.ItemID
			require.Truef(t, added[id], "delta before added for %s", id)
			require.Falsef(t, done[id], "delta after done for %s", id)
			deltaText[id] += e.Delta
		case "response.function_call_arguments.delta":
			id := e.ItemID
			require.Truef(t, added[id], "delta before added for %s", id)
			require.Falsef(t, done[id], "delta after done for %s", id)
			deltaText[id] += e.Delta
		case "response.output_item.done":
			require.NotNil(t, e.Item)
			require.Truef(t, added[e.Item.ID], "done before added for %s", e.Item.ID)
			require.Falsef(t, done[e.Item.ID], "duplicate done for %s", e.Item.ID)
			done[e.Item.ID] = true
			doneOrder = append(doneOrder, e.Item.ID)
			indexOrder = append(indexOrder, e.OutputIndex)
			doneType[e.Item.ID] = e.Item.Type
			switch e.Item.Type {
			case "reasoning":
				require.NotEmpty(t, e.Item.Summary)
				doneText[e.Item.ID] = e.Item.Summary[0].Text
			case "message":
				require.NotEmpty(t, e.Item.Content)
				doneText[e.Item.ID] = e.Item.Content[0].Text
			case "function_call":
				doneText[e.Item.ID] = e.Item.Arguments
			}
		case "response.completed", "response.incomplete":
			require.NotNil(t, e.Response)
			require.Len(t, e.Response.Output, len(doneOrder), "terminal output length must equal delivered item count")
			for i, out := range e.Response.Output {
				require.Equal(t, doneOrder[i], out.ID, "terminal output order/id must match done order")
				require.Equal(t, doneType[out.ID], out.Type)
				switch out.Type {
				case "reasoning":
					require.Equal(t, doneText[out.ID], out.Summary[0].Text)
				case "message":
					require.Equal(t, doneText[out.ID], out.Content[0].Text)
				case "function_call":
					require.Equal(t, doneText[out.ID], out.Arguments)
				}
			}
		}
	}
	require.Equal(t, indexOrder, outputIndices(events))
	require.Equal(t, nextIndex, len(doneOrder), "every added item must be done")
	for id, txt := range deltaText {
		if doneType[id] != "" {
			require.Equalf(t, doneText[id], txt, "accumulated deltas must equal done content for %s", id)
		}
	}
}

// 连续 reasoning 段（分阶段产生相邻 reasoning）在下一轮 CC 历史中必须保留顺序，
// 不能被简单赋值覆盖；真实字符串必须两段都在。
func TestResponsesToChat_ConsecutiveReasoningItemsPreserveOrder(t *testing.T) {
	req := &ResponsesRequest{
		Model: "deepseek-reasoner",
		Input: json.RawMessage(`[
			{"type":"reasoning","summary":[{"type":"summary_text","text":"segment A"}]},
			{"type":"reasoning","summary":[{"type":"summary_text","text":"segment B"}]},
			{"type":"message","role":"assistant","content":[{"type":"output_text","text":"final"}]}
		]`),
	}
	out, err := ResponsesToChatCompletionsRequest(req)
	require.NoError(t, err)
	require.Len(t, out.Messages, 1)
	require.Equal(t, "assistant", out.Messages[0].Role)
	require.Equal(t, "segment A\nsegment B", out.Messages[0].ReasoningContent,
		"consecutive reasoning segments must both survive in order")
}

// 相邻 reasoning 段后接工具调用时，两段 reasoning 都必须回放到该 assistant 消息。
func TestResponsesToChat_ConsecutiveReasoningBeforeToolCallPreserved(t *testing.T) {
	req := &ResponsesRequest{
		Model: "deepseek-reasoner",
		Input: json.RawMessage(`[
			{"type":"reasoning","summary":[{"type":"summary_text","text":"A"}]},
			{"type":"reasoning","summary":[{"type":"summary_text","text":"B"}]},
			{"type":"function_call","call_id":"call_1","name":"exec","arguments":"{}"},
			{"type":"function_call_output","call_id":"call_1","output":"ok"}
		]`),
	}
	out, err := ResponsesToChatCompletionsRequest(req)
	require.NoError(t, err)
	require.Len(t, out.Messages, 2)
	require.Equal(t, "assistant", out.Messages[0].Role)
	require.Equal(t, "A\nB", out.Messages[0].ReasoningContent)
	require.Equal(t, "call_1", out.Messages[0].ToolCalls[0].ID)
}

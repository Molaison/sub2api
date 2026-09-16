package service

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/Wei-Shaw/sub2api/internal/config"
	"github.com/Wei-Shaw/sub2api/internal/pkg/openai_compat"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
	"github.com/tidwall/gjson"
)

func TestHandleStreamingResponsePassthroughDeduplicatesFunctionCallArguments(t *testing.T) {
	gin.SetMode(gin.TestMode)

	argsA := `{"cmd":"echo hi","meta":{"nested":[1,{"ok":true}],"quote":"a}b"}}`
	argsB := `{"path":"/tmp/file","patch":{"ops":[{"op":"replace","value":{"lines":["x","y"]}}]}}`
	upstreamBody := strings.Join([]string{
		passthroughSSEData(`{"type":"response.created","response":{"id":"resp_passthrough_args","model":"gpt-5.4"}}`),
		passthroughSSEData(`{"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","id":"fc_a","call_id":"call_a","name":"exec_command","arguments":"","status":"in_progress"}}`),
		passthroughSSEData(functionArgsDeltaJSON(0, "fc_a", "call_a", "exec_command", `{"cmd":`)),
		passthroughSSEData(functionArgsDeltaJSON(0, "fc_a", "call_a", "exec_command", `"echo hi","meta":{"nested":[1,{"ok":true}],"quote":"a}b"}}`)),
		passthroughSSEData(functionArgsDoneJSON(0, "fc_a", "call_a", "exec_command", argsA+argsA)),
		passthroughSSEData(outputItemDoneJSON(0, "fc_a", "call_a", "exec_command", argsA+argsA)),
		passthroughSSEData(`{"type":"response.output_item.added","output_index":1,"item":{"type":"function_call","id":"fc_b","call_id":"call_b","name":"apply_patch","arguments":"","status":"in_progress"}}`),
		passthroughSSEData(functionArgsDeltaJSON(1, "fc_b", "call_b", "apply_patch", `{"path":"/tmp/file",`)),
		passthroughSSEData(functionArgsDeltaJSON(1, "fc_b", "call_b", "apply_patch", `"patch":{"ops":[{"op":"replace","value":{"lines":["x","y"]}}]}}`)),
		passthroughSSEData(functionArgsDoneJSON(1, "fc_b", "call_b", "apply_patch", argsB+argsB)),
		passthroughSSEData(outputItemDoneJSON(1, "fc_b", "call_b", "apply_patch", argsB+argsB)),
		passthroughSSEData(completedWithFunctionCallsJSON(argsA+argsA, argsB+argsB)),
		"data: [DONE]\n\n",
	}, "")

	rec := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(rec)
	c.Request = httptest.NewRequest(http.MethodPost, "/v1/responses", nil)
	resp := &http.Response{
		StatusCode: http.StatusOK,
		Header:     http.Header{"Content-Type": []string{"text/event-stream"}},
		Body:       io.NopCloser(strings.NewReader(upstreamBody)),
	}

	svc := &OpenAIGatewayService{}
	result, err := svc.handleStreamingResponsePassthrough(context.Background(), resp, c, &Account{ID: 1}, time.Now(), "gpt-5.4", "gpt-5.4")
	require.NoError(t, err)
	require.NotNil(t, result)

	events := collectSSEDataPayloads(t, rec.Body.String())
	require.Equal(t, argsA, accumulateFunctionArgumentDeltas(events, "call_a"))
	require.Equal(t, argsB, accumulateFunctionArgumentDeltas(events, "call_b"))

	require.Equal(t, argsA, gjson.Get(findSSEEvent(t, events, "response.function_call_arguments.done", "call_a"), "arguments").String())
	require.Equal(t, argsB, gjson.Get(findSSEEvent(t, events, "response.function_call_arguments.done", "call_b"), "arguments").String())
	require.Equal(t, argsA, gjson.Get(findSSEEvent(t, events, "response.output_item.done", "call_a"), "item.arguments").String())
	require.Equal(t, argsB, gjson.Get(findSSEEvent(t, events, "response.output_item.done", "call_b"), "item.arguments").String())

	completed := findSSEEvent(t, events, "response.completed", "")
	require.Equal(t, argsA, gjson.Get(completed, "response.output.0.arguments").String())
	require.Equal(t, argsB, gjson.Get(completed, "response.output.1.arguments").String())
	requireJSONArgument(t, gjson.Get(completed, "response.output.0.arguments").String())
	requireJSONArgument(t, gjson.Get(completed, "response.output.1.arguments").String())
}

func TestForwardResponsesChatCompletionsFallbackKeepsFunctionArgumentsSingle(t *testing.T) {
	gin.SetMode(gin.TestMode)

	body := []byte(`{"model":"gpt-5.4","input":"run a command","stream":true}`)
	rec := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(rec)
	c.Request = httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(string(body)))
	c.Request.Header.Set("Content-Type", "application/json")

	upstreamBody := strings.Join([]string{
		passthroughSSEData(chatToolCallChunkJSON(true, "")),
		"",
		passthroughSSEData(chatToolCallChunkJSON(false, `{"cmd":"echo hi"}`)),
		"",
		`data: {"id":"chatcmpl_tool","object":"chat.completion.chunk","model":"gpt-5.4","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}}`,
		"",
		"data: [DONE]",
		"",
	}, "\n")
	upstream := &httpUpstreamRecorder{resp: &http.Response{
		StatusCode: http.StatusOK,
		Header:     http.Header{"Content-Type": []string{"text/event-stream"}, "x-request-id": []string{"rid_fallback_tool_args"}},
		Body:       io.NopCloser(strings.NewReader(upstreamBody)),
	}}
	account := passthroughArgsFallbackAccount()
	account.Extra = map[string]any{
		openai_compat.ExtraKeyResponsesMode: string(openai_compat.ResponsesSupportModeForceChatCompletions),
	}
	svc := &OpenAIGatewayService{
		cfg:          passthroughArgsTestConfig(),
		httpUpstream: upstream,
	}

	result, err := svc.Forward(context.Background(), c, account, body)
	require.NoError(t, err)
	require.NotNil(t, result)

	const wantArgs = `{"cmd":"echo hi"}`
	events := collectSSEDataPayloads(t, rec.Body.String())
	require.Equal(t, wantArgs, accumulateFunctionArgumentDeltas(events, "chatcmpl-tool-a"))
	require.Equal(t, wantArgs, gjson.Get(findSSEEvent(t, events, "response.function_call_arguments.done", "chatcmpl-tool-a"), "arguments").String())
	require.Equal(t, wantArgs, gjson.Get(findSSEEvent(t, events, "response.output_item.done", "chatcmpl-tool-a"), "item.arguments").String())
}

func TestNeedsOpenAIResponsesClientToolAdaptationTypedLocations(t *testing.T) {
	for _, body := range []string{
		`{"tools":[{"type":"namespace","name":"collaboration","tools":[{"type":"function","name":"spawn_agent"}]}]}`,
		`{"input":[{"type":"additional_tools","tools":[{"type":"namespace","name":"collaboration","tools":[]}]}]}`,
		`{"input":[{"type":"function_call","namespace":"collaboration","name":"spawn_agent","call_id":"call_1"}]}`,
	} {
		require.True(t, needsOpenAIResponsesClientToolAdaptation([]byte(body)))
	}
	require.False(t, needsOpenAIResponsesClientToolAdaptation([]byte(
		`{"tools":[{"type":"function","name":"submit","parameters":{"type":"object","properties":{"kind":{"const":"namespace"}}}}]}`,
	)))
}

func TestForwardParateraDeepSeekAdaptsNamespaceAndPreservesReplay(t *testing.T) {
	gin.SetMode(gin.TestMode)
	body := []byte(`{"model":"DeepSeek-V4.1-Flash","store":false,"stream":false,"tools":[` +
		`{"type":"namespace","name":"collaboration","tools":[{"type":"function","name":"spawn_agent","parameters":{"type":"object"}}]}` +
		`],"input":[` +
		`{"type":"message","role":"user","content":"delegate"},` +
		`{"type":"reasoning","summary":[],"content":[{"type":"reasoning_text","text":"keep reasoning"}]},` +
		`{"type":"function_call","namespace":"collaboration","name":"spawn_agent","call_id":"call_old","arguments":"{}"},` +
		`{"type":"message","role":"assistant","content":[{"type":"output_text","text":"keep preface"}]},` +
		`{"type":"function_call_output","call_id":"call_old","output":"child-old"}` +
		`]}`)
	rec := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(rec)
	c.Request = httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(string(body)))
	c.Request.Header.Set("Content-Type", "application/json")
	upstream := &httpUpstreamRecorder{resp: &http.Response{
		StatusCode: http.StatusOK,
		Header:     http.Header{"Content-Type": []string{"application/json"}},
		Body: io.NopCloser(strings.NewReader(
			`{"id":"resp_1","status":"completed","model":"DeepSeek-V4.1-Flash","output":[{"type":"function_call","id":"fc_new","call_id":"call_new","name":"collaboration__spawn_agent","arguments":"{}"}],"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}`,
		)),
	}}
	account := &Account{
		ID:       120,
		Name:     "paratera-deepseek-test",
		Platform: PlatformOpenAI,
		Type:     AccountTypeAPIKey,
		Credentials: map[string]any{
			"api_key":  "sk-test",
			"base_url": "https://llmapi.paratera.com/v1",
		},
		Extra: map[string]any{
			openai_compat.ExtraKeyResponsesMode: string(openai_compat.ResponsesSupportModeForceResponses),
		},
	}
	svc := &OpenAIGatewayService{cfg: passthroughArgsTestConfig(), httpUpstream: upstream}

	result, err := svc.Forward(context.Background(), c, account, body)
	require.NoError(t, err)
	require.NotNil(t, result)
	require.Equal(t, "function", gjson.GetBytes(upstream.lastBody, "tools.0.type").String())
	require.Equal(t, "collaboration__spawn_agent", gjson.GetBytes(upstream.lastBody, "tools.0.name").String())
	require.Equal(t, "keep reasoning", gjson.GetBytes(upstream.lastBody, "input.1.content.0.text").String())
	require.Equal(t, "message", gjson.GetBytes(upstream.lastBody, "input.2.type").String())
	require.Equal(t, "keep preface", gjson.GetBytes(upstream.lastBody, "input.2.content.0.text").String())
	require.Equal(t, "collaboration__spawn_agent", gjson.GetBytes(upstream.lastBody, "input.3.name").String())
	require.Equal(t, "function_call_output", gjson.GetBytes(upstream.lastBody, "input.4.type").String())
	require.Equal(t, "collaboration", gjson.Get(rec.Body.String(), "output.0.namespace").String())
	require.Equal(t, "spawn_agent", gjson.Get(rec.Body.String(), "output.0.name").String())
}

func passthroughSSEData(payload string) string {
	return "data: " + payload + "\n\n"
}

func functionArgsDeltaJSON(outputIndex int, itemID, callID, name, delta string) string {
	return fmt.Sprintf(
		`{"type":"response.function_call_arguments.delta","output_index":%d,"item_id":%s,"call_id":%s,"name":%s,"delta":%s}`,
		outputIndex,
		strconv.Quote(itemID),
		strconv.Quote(callID),
		strconv.Quote(name),
		strconv.Quote(delta),
	)
}

func functionArgsDoneJSON(outputIndex int, itemID, callID, name, arguments string) string {
	return fmt.Sprintf(
		`{"type":"response.function_call_arguments.done","output_index":%d,"item_id":%s,"call_id":%s,"name":%s,"arguments":%s}`,
		outputIndex,
		strconv.Quote(itemID),
		strconv.Quote(callID),
		strconv.Quote(name),
		strconv.Quote(arguments),
	)
}

func outputItemDoneJSON(outputIndex int, itemID, callID, name, arguments string) string {
	return fmt.Sprintf(
		`{"type":"response.output_item.done","output_index":%d,"item":{"type":"function_call","id":%s,"call_id":%s,"name":%s,"arguments":%s,"status":"completed"}}`,
		outputIndex,
		strconv.Quote(itemID),
		strconv.Quote(callID),
		strconv.Quote(name),
		strconv.Quote(arguments),
	)
}

func completedWithFunctionCallsJSON(argsA, argsB string) string {
	return fmt.Sprintf(
		`{"type":"response.completed","response":{"id":"resp_passthrough_args","status":"completed","output":[{"type":"function_call","id":"fc_a","call_id":"call_a","name":"exec_command","arguments":%s,"status":"completed"},{"type":"function_call","id":"fc_b","call_id":"call_b","name":"apply_patch","arguments":%s,"status":"completed"}],"usage":{"input_tokens":2,"output_tokens":3,"total_tokens":5}}}`,
		strconv.Quote(argsA),
		strconv.Quote(argsB),
	)
}

func chatToolCallChunkJSON(includeIdentity bool, arguments string) string {
	identity := ""
	functionFields := make([]string, 0, 2)
	if includeIdentity {
		identity = `"id":"chatcmpl-tool-a","type":"function",`
		functionFields = append(functionFields, `"name":"exec_command"`)
	}
	if includeIdentity || arguments != "" {
		functionFields = append(functionFields, `"arguments":`+strconv.Quote(arguments))
	}
	return fmt.Sprintf(
		`{"id":"chatcmpl_tool","object":"chat.completion.chunk","model":"gpt-5.4","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,%s"function":{%s}}]},"finish_reason":null}]}`,
		identity,
		strings.Join(functionFields, ","),
	)
}

func passthroughArgsTestConfig() *config.Config {
	return &config.Config{
		Security: config.SecurityConfig{
			URLAllowlist: config.URLAllowlistConfig{
				Enabled:           false,
				AllowInsecureHTTP: true,
			},
		},
	}
}

func passthroughArgsFallbackAccount() *Account {
	return &Account{
		ID:          102,
		Name:        "passthrough-args-openai-apikey",
		Platform:    PlatformOpenAI,
		Type:        AccountTypeAPIKey,
		Concurrency: 1,
		Credentials: map[string]any{
			"api_key":  "sk-test",
			"base_url": "http://upstream.example",
		},
	}
}

func collectSSEDataPayloads(t *testing.T, body string) []string {
	t.Helper()
	scanner := bufio.NewScanner(strings.NewReader(body))
	var events []string
	for scanner.Scan() {
		data, ok := extractOpenAISSEDataLine(scanner.Text())
		if !ok {
			continue
		}
		if strings.TrimSpace(data) == "[DONE]" {
			continue
		}
		require.True(t, gjson.Valid(data), "invalid SSE data payload: %s", data)
		events = append(events, data)
	}
	require.NoError(t, scanner.Err())
	return events
}

func findSSEEvent(t *testing.T, events []string, eventType, callID string) string {
	t.Helper()
	for _, event := range events {
		if gjson.Get(event, "type").String() != eventType {
			continue
		}
		if callID == "" ||
			gjson.Get(event, "call_id").String() == callID ||
			gjson.Get(event, "item.call_id").String() == callID {
			return event
		}
	}
	t.Fatalf("missing event type=%s call_id=%s in %d events", eventType, callID, len(events))
	return ""
}

func accumulateFunctionArgumentDeltas(events []string, callID string) string {
	var b strings.Builder
	for _, event := range events {
		if gjson.Get(event, "type").String() != "response.function_call_arguments.delta" {
			continue
		}
		if gjson.Get(event, "call_id").String() != callID {
			continue
		}
		_, _ = b.WriteString(gjson.Get(event, "delta").String())
	}
	return b.String()
}

func requireJSONArgument(t *testing.T, arguments string) {
	t.Helper()
	var decoded any
	require.NoError(t, json.Unmarshal([]byte(arguments), &decoded))
}

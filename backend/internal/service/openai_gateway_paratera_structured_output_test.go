package service

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/require"
	"github.com/tidwall/gjson"
)

func TestPrepareParateraDeepSeekStructuredOutput(t *testing.T) {
	body := []byte(`{
		"model":"DeepSeek-V4.1-Flash",
		"instructions":"Keep existing instructions.",
		"input":"return an object",
		"stream":true,
		"text":{"format":{"type":"json_schema","name":"answer","strict":true,"schema":{"type":"object","properties":{"result":{"type":"string"}},"required":["result"],"additionalProperties":false}}}
	}`)

	bridged, bridge, changed, err := prepareParateraDeepSeekStructuredOutput(body)
	require.NoError(t, err)
	require.True(t, changed)
	require.NotNil(t, bridge)
	require.False(t, gjson.GetBytes(bridged, "text.format").Exists())
	require.False(t, gjson.GetBytes(bridged, "response_format").Exists())
	instructions := gjson.GetBytes(bridged, "instructions").String()
	require.Contains(t, instructions, "Keep existing instructions.")
	require.Contains(t, instructions, parateraStructuredOutputInstruction)
	require.Contains(t, instructions, `<json_schema>{"additionalProperties":false`)

	validSSE := []byte("data: {\"type\":\"response.completed\",\"response\":{\"output\":[{\"type\":\"message\",\"content\":[{\"type\":\"output_text\",\"text\":\"{\\\"result\\\":\\\"ok\\\"}\"}]}]}}\n\ndata: [DONE]\n\n")
	require.NoError(t, validateParateraDeepSeekStructuredOutputBody(validSSE, bridge))

	invalidSSE := []byte("data: {\"type\":\"response.completed\",\"response\":{\"output\":[{\"type\":\"message\",\"content\":[{\"type\":\"output_text\",\"text\":\"{\\\"result\\\":7}\"}]}]}}\n\ndata: [DONE]\n\n")
	err = validateParateraDeepSeekStructuredOutputBody(invalidSSE, bridge)
	require.ErrorContains(t, err, "does not match requested JSON Schema")
}

func TestPrepareParateraDeepSeekStructuredOutputIgnoresOtherFormats(t *testing.T) {
	body := []byte(`{"model":"DeepSeek-V4.1-Flash","text":{"format":{"type":"json_object"}}}`)
	bridged, bridge, changed, err := prepareParateraDeepSeekStructuredOutput(body)
	require.NoError(t, err)
	require.False(t, changed)
	require.Nil(t, bridge)
	require.Equal(t, string(body), string(bridged))
}

func TestValidateParateraDeepSeekStructuredOutputRejectsTrailingText(t *testing.T) {
	body := []byte(`{"text":{"format":{"type":"json_schema","schema":{"type":"object"}}}}`)
	_, bridge, changed, err := prepareParateraDeepSeekStructuredOutput(body)
	require.NoError(t, err)
	require.True(t, changed)

	response := []byte(`{"output":[{"type":"message","content":[{"type":"output_text","text":"{} trailing"}]}]}`)
	err = validateParateraDeepSeekStructuredOutputBody(response, bridge)
	require.Error(t, err)
	require.True(t, strings.Contains(err.Error(), "trailing") || strings.Contains(err.Error(), "more than one"))
}

func TestPrepareParateraDeepSeekStructuredOutputReferencePolicy(t *testing.T) {
	localRef := []byte(`{"text":{"format":{"type":"json_schema","schema":{"$defs":{"value":{"type":"string"}},"type":"object","properties":{"result":{"$ref":"#/$defs/value"}}}}}}`)
	_, bridge, changed, err := prepareParateraDeepSeekStructuredOutput(localRef)
	require.NoError(t, err)
	require.True(t, changed)
	require.NotNil(t, bridge)

	externalRef := []byte(`{"text":{"format":{"type":"json_schema","schema":{"$ref":"https://example.invalid/schema.json"}}}}`)
	_, bridge, changed, err = prepareParateraDeepSeekStructuredOutput(externalRef)
	require.ErrorContains(t, err, "external JSON Schema references are disabled")
	require.False(t, changed)
	require.Nil(t, bridge)
}

func TestValidateParateraDeepSeekStructuredOutputAllowsToolRound(t *testing.T) {
	body := []byte(`{"text":{"format":{"type":"json_schema","schema":{"type":"object","required":["result"]}}}}`)
	_, bridge, changed, err := prepareParateraDeepSeekStructuredOutput(body)
	require.NoError(t, err)
	require.True(t, changed)

	toolRound := []byte("data: {\"type\":\"response.completed\",\"response\":{\"output\":[{\"type\":\"message\",\"content\":[{\"type\":\"output_text\",\"text\":\"\"}]},{\"type\":\"function_call\",\"name\":\"exec_command\",\"arguments\":\"{\\\"cmd\\\":\\\"printf marker\\\"}\"}]}}\n\ndata: [DONE]\n\n")
	require.NoError(t, validateParateraDeepSeekStructuredOutputBody(toolRound, bridge))
}

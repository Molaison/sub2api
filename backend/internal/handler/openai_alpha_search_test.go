package handler

import (
	"testing"

	"github.com/Wei-Shaw/sub2api/internal/config"
	"github.com/stretchr/testify/require"
)

func TestAlphaSearchExecutionModelExplicitDelegation(t *testing.T) {
	h := &OpenAIGatewayHandler{cfg: &config.Config{}}
	const conversation = "DeepSeek-V4.1-Flash"
	require.Equal(t, conversation, h.alphaSearchExecutionModel(conversation))
	h.cfg.Gateway.OpenAIAlphaSearchModel = "gpt-5.6-sol"
	// A backend alone must not redirect all models or implicitly enable delegation.
	require.Equal(t, conversation, h.alphaSearchExecutionModel(conversation))
	h.cfg.Gateway.OpenAIAlphaSearchDelegatedModels = []string{" " + conversation + " "}
	require.Equal(t, "gpt-5.6-sol", h.alphaSearchExecutionModel(conversation))
	require.Equal(t, "DeepSeek-V4.1-Flash-other", h.alphaSearchExecutionModel("DeepSeek-V4.1-Flash-other"))
	require.Equal(t, "gpt-6-astra", h.alphaSearchExecutionModel("gpt-6-astra"))
	h.cfg.Gateway.OpenAIAlphaSearchModel = " "
	require.Equal(t, conversation, h.alphaSearchExecutionModel(conversation))
	var absent *OpenAIGatewayHandler
	require.Equal(t, conversation, absent.alphaSearchExecutionModel(conversation))
}

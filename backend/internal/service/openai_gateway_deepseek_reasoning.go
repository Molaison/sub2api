package service

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/Wei-Shaw/sub2api/internal/pkg/openai"
	"github.com/gin-gonic/gin"
	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
)

const deepSeekCodexReasoningHeader = "**DeepSeek reasoning**\n\n"

type deepSeekCodexReasoningSummaryKey struct {
	itemID       string
	outputIndex  int64
	summaryIndex int64
}

type deepSeekCodexReasoningStreamDecorator struct {
	started map[deepSeekCodexReasoningSummaryKey]struct{}
}

func shouldUseDeepSeekCodexReasoningCompatibility(account *Account, isCodexCLI bool) bool {
	return isCodexCLI && account != nil && account.IsDeepseek() && account.UsesNativeCNResponses()
}

func (s *OpenAIGatewayService) shouldDecorateDeepSeekCodexReasoning(c *gin.Context, account *Account) bool {
	isCodexCLI := s != nil && s.cfg != nil && s.cfg.Gateway.ForceCodexCLI
	if c != nil {
		isCodexCLI = isCodexCLI || openai.IsCodexOfficialClientByHeaders(
			c.GetHeader("User-Agent"),
			c.GetHeader("originator"),
		)
	}
	return shouldUseDeepSeekCodexReasoningCompatibility(account, isCodexCLI)
}

func newDeepSeekCodexReasoningStreamDecorator(enabled bool) *deepSeekCodexReasoningStreamDecorator {
	if !enabled {
		return nil
	}
	return &deepSeekCodexReasoningStreamDecorator{
		started: make(map[deepSeekCodexReasoningSummaryKey]struct{}),
	}
}

func (d *deepSeekCodexReasoningStreamDecorator) Decorate(data []byte, eventType string) ([]byte, bool) {
	if d == nil {
		return data, false
	}
	if eventType == "" {
		eventType = strings.TrimSpace(gjson.GetBytes(data, "type").String())
	}
	switch eventType {
	case "response.reasoning_summary_text.delta":
		delta := gjson.GetBytes(data, "delta")
		if delta.Type != gjson.String || strings.TrimSpace(delta.String()) == "" {
			return data, false
		}
		key := deepSeekCodexReasoningSummaryKey{
			itemID:       gjson.GetBytes(data, "item_id").String(),
			outputIndex:  gjson.GetBytes(data, "output_index").Int(),
			summaryIndex: gjson.GetBytes(data, "summary_index").Int(),
		}
		if _, seen := d.started[key]; seen {
			return data, false
		}
		decorated, changed := decorateDeepSeekReasoningTextAtPath(data, "delta")
		if changed || strings.HasPrefix(delta.String(), deepSeekCodexReasoningHeader) {
			d.started[key] = struct{}{}
		}
		return decorated, changed
	case "response.reasoning_summary_text.done":
		return decorateDeepSeekReasoningTextAtPath(data, "text")
	case "response.reasoning_summary_part.done":
		if strings.TrimSpace(gjson.GetBytes(data, "part.type").String()) != "summary_text" {
			return data, false
		}
		return decorateDeepSeekReasoningTextAtPath(data, "part.text")
	case "response.output_item.done":
		return decorateDeepSeekReasoningItemAtPath(data, "item")
	case "response.completed", "response.done":
		return decorateDeepSeekReasoningItemsAtPath(data, "response.output")
	default:
		return data, false
	}
}

func (s *OpenAIGatewayService) decorateDeepSeekCodexReasoningResponse(c *gin.Context, account *Account, body []byte) []byte {
	if !s.shouldDecorateDeepSeekCodexReasoning(c, account) {
		return body
	}
	decorated, _ := decorateDeepSeekReasoningItemsAtPath(body, "output")
	return decorated
}

func decorateDeepSeekReasoningItemAtPath(data []byte, itemPath string) ([]byte, bool) {
	item := gjson.GetBytes(data, itemPath)
	if !item.IsObject() || strings.TrimSpace(item.Get("type").String()) != "reasoning" {
		return data, false
	}
	return decorateDeepSeekReasoningSummaryAtPath(data, itemPath+".summary")
}

func decorateDeepSeekReasoningItemsAtPath(data []byte, outputPath string) ([]byte, bool) {
	outputs := gjson.GetBytes(data, outputPath)
	if !outputs.IsArray() {
		return data, false
	}
	updated := data
	changed := false
	for i, output := range outputs.Array() {
		if strings.TrimSpace(output.Get("type").String()) != "reasoning" {
			continue
		}
		var itemChanged bool
		updated, itemChanged = decorateDeepSeekReasoningSummaryAtPath(
			updated,
			outputPath+"."+strconv.Itoa(i)+".summary",
		)
		changed = changed || itemChanged
	}
	return updated, changed
}

func decorateDeepSeekReasoningSummaryAtPath(data []byte, summaryPath string) ([]byte, bool) {
	summaries := gjson.GetBytes(data, summaryPath)
	if !summaries.IsArray() {
		return data, false
	}
	updated := data
	changed := false
	for i, summary := range summaries.Array() {
		if strings.TrimSpace(summary.Get("type").String()) != "summary_text" {
			continue
		}
		var textChanged bool
		updated, textChanged = decorateDeepSeekReasoningTextAtPath(
			updated,
			summaryPath+"."+strconv.Itoa(i)+".text",
		)
		changed = changed || textChanged
	}
	return updated, changed
}

func decorateDeepSeekReasoningTextAtPath(data []byte, path string) ([]byte, bool) {
	value := gjson.GetBytes(data, path)
	if value.Type != gjson.String {
		return data, false
	}
	text := value.String()
	if strings.TrimSpace(text) == "" || strings.HasPrefix(text, deepSeekCodexReasoningHeader) {
		return data, false
	}
	updated, err := sjson.SetBytes(data, path, deepSeekCodexReasoningHeader+text)
	if err != nil {
		return data, false
	}
	return updated, true
}

func stripDeepSeekCodexReasoningHeaders(body []byte) ([]byte, bool, error) {
	input := gjson.GetBytes(body, "input")
	if !input.IsArray() {
		return body, false, nil
	}
	updated := body
	changed := false
	for itemIndex, item := range input.Array() {
		if strings.TrimSpace(item.Get("type").String()) != "reasoning" {
			continue
		}
		for summaryIndex, summary := range item.Get("summary").Array() {
			if strings.TrimSpace(summary.Get("type").String()) != "summary_text" {
				continue
			}
			text := summary.Get("text")
			if text.Type != gjson.String || !strings.HasPrefix(text.String(), deepSeekCodexReasoningHeader) {
				continue
			}
			path := "input." + strconv.Itoa(itemIndex) + ".summary." + strconv.Itoa(summaryIndex) + ".text"
			var err error
			updated, err = sjson.SetBytes(updated, path, strings.TrimPrefix(text.String(), deepSeekCodexReasoningHeader))
			if err != nil {
				return body, false, fmt.Errorf("strip DeepSeek Codex reasoning header: %w", err)
			}
			changed = true
		}
	}
	return updated, changed, nil
}

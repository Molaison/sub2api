package service

import (
	"encoding/json"
	"fmt"
	"sort"
	"strings"

	"github.com/google/uuid"
	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
)

// Paratera consumes reasoning.content, but Codex replays reasoning.summary.
// Keep this conversion at the provider boundary: OpenAI must never receive
// Paratera's output_text content inside a reasoning item.
func normalizeParateraDeepSeekReasoningReplay(body []byte) ([]byte, bool, error) {
	input := gjson.GetBytes(body, "input")
	if !input.IsArray() {
		return body, false, nil
	}
	items := input.Array()
	rawItems := make([]string, len(items))
	changed := false
	for i, item := range items {
		rawItems[i] = item.Raw
		if item.Get("type").String() != "reasoning" {
			continue
		}
		content := item.Get("content")
		if len(content.Array()) > 0 {
			continue // Already in the upstream's native representation.
		}
		parts := make([]map[string]string, 0)
		for _, part := range item.Get("summary").Array() {
			if part.Get("type").String() == "summary_text" {
				text := strings.TrimPrefix(part.Get("text").String(), deepSeekCodexReasoningHeader)
				if text != "" {
					parts = append(parts, map[string]string{"type": "output_text", "text": text})
				}
			}
		}
		if len(parts) == 0 {
			continue
		}
		updated, err := sjson.Set(item.Raw, "content", parts)
		if err != nil {
			return body, false, err
		}
		rawItems[i] = updated
		changed = true
	}
	if !changed {
		return body, false, nil
	}
	return replaceOpenAIRawInput(body, input, rawItems), true, nil
}

// A DeepSeek completion has one reasoning phase before its message/tool calls.
// The upstream may use a new id for every delta, omit the entire item lifecycle,
// or put the only complete reasoning in response.completed.output.content.
// Codex persists output_item.done, not the terminal response's output array.
type parateraReasoningState struct {
	id      string
	index   int64
	parts   map[int64]*strings.Builder
	started map[int64]bool
	done    bool
}

func (s *parateraReasoningState) setPart(index int64, text string) {
	builder := &strings.Builder{}
	builder.WriteString(text)
	s.parts[index] = builder
}

func (r *parateraDeepSeekItemLifecycleRepair) openReasoning(id string, index int64) [][]byte {
	if r.reasoning != nil {
		return nil
	}
	if id == "" {
		id = "rs_" + uuid.NewString()
	}
	r.reasoning = &parateraReasoningState{id: id, index: index, parts: make(map[int64]*strings.Builder), started: make(map[int64]bool)}
	return [][]byte{r.reasoningEvent("response.output_item.added", map[string]any{
		"item": map[string]any{"type": "reasoning", "id": id, "status": "in_progress", "summary": []any{}},
	})}
}

func (r *parateraDeepSeekItemLifecycleRepair) reasoningEvent(kind string, fields map[string]any) []byte {
	fields["type"] = kind
	fields["output_index"] = r.reasoning.index
	if r.model != "" {
		fields["model"] = r.model
	}
	encoded, _ := json.Marshal(fields) // Only JSON primitives are constructed here.
	return encoded
}

func (r *parateraDeepSeekItemLifecycleRepair) reasoningItem() map[string]any {
	s := r.reasoning
	indices := make([]int64, 0, len(s.parts))
	for index := range s.parts {
		indices = append(indices, index)
	}
	sort.Slice(indices, func(i, j int) bool { return indices[i] < indices[j] })
	parts := make([]any, 0, len(indices))
	for _, index := range indices {
		parts = append(parts, map[string]any{"type": "summary_text", "text": s.parts[index].String()})
	}
	return map[string]any{"type": "reasoning", "id": s.id, "status": "completed", "summary": parts}
}

func (r *parateraDeepSeekItemLifecycleRepair) finishReasoning() [][]byte {
	s := r.reasoning
	if s == nil || s.done {
		return nil
	}
	s.done = true
	item := r.reasoningItem()
	events := make([][]byte, 0)
	indices := make([]int64, 0, len(s.parts))
	for index := range s.parts {
		indices = append(indices, index)
	}
	sort.Slice(indices, func(i, j int) bool { return indices[i] < indices[j] })
	for _, index := range indices {
		text := s.parts[index].String()
		events = append(events,
			r.reasoningEvent("response.reasoning_summary_text.done", map[string]any{"item_id": s.id, "summary_index": index, "text": text}),
			r.reasoningEvent("response.reasoning_summary_part.done", map[string]any{"item_id": s.id, "summary_index": index, "part": map[string]any{"type": "summary_text", "text": text}}))
	}
	return append(events, r.reasoningEvent("response.output_item.done", map[string]any{"item": item}))
}

func (r *parateraDeepSeekItemLifecycleRepair) readReasoningItem(item gjson.Result) {
	if r.reasoning.done {
		return
	}
	parts := item.Get("summary").Array()
	if len(parts) == 0 {
		parts = item.Get("content").Array()
	}
	if len(parts) == 0 {
		return
	}
	r.reasoning.parts = make(map[int64]*strings.Builder)
	for i, part := range parts {
		r.reasoning.setPart(int64(i), part.Get("text").String())
	}
}

func (r *parateraDeepSeekItemLifecycleRepair) repairReasoning(payload []byte, kind string) (repairOutcome, bool) {
	index := gjson.GetBytes(payload, "output_index").Int()
	item := gjson.GetBytes(payload, "item")
	if (kind == "response.output_item.added" || kind == "response.output_item.done") && item.Get("type").String() == "reasoning" {
		prefix := r.openReasoning(item.Get("id").String(), index)
		if kind == "response.output_item.done" {
			r.readReasoningItem(item)
			prefix = append(prefix, r.finishReasoning()...)
		}
		return repairOutcome{prefix: prefix, drop: true}, true
	}
	if strings.HasPrefix(kind, "response.reasoning_summary_") {
		prefix := r.openReasoning("", index)
		s := r.reasoning
		if s.done {
			return repairOutcome{drop: true}, true
		}
		partIndex := gjson.GetBytes(payload, "summary_index").Int()
		if !s.started[partIndex] {
			s.started[partIndex] = true
			prefix = append(prefix, r.reasoningEvent("response.reasoning_summary_part.added", map[string]any{
				"item_id": s.id, "summary_index": partIndex, "part": map[string]any{"type": "summary_text", "text": ""},
			}))
		}
		switch kind {
		case "response.reasoning_summary_text.delta":
			if s.parts[partIndex] == nil {
				s.parts[partIndex] = &strings.Builder{}
			}
			s.parts[partIndex].WriteString(gjson.GetBytes(payload, "delta").String())
			patched, _ := sjson.SetBytes(payload, "item_id", s.id)
			patched, _ = sjson.SetBytes(patched, "summary_index", partIndex)
			patched, _ = sjson.SetBytes(patched, "output_index", s.index)
			return repairOutcome{prefix: prefix, replacement: patched}, true
		case "response.reasoning_summary_text.done":
			s.setPart(partIndex, gjson.GetBytes(payload, "text").String())
		case "response.reasoning_summary_part.done":
			s.setPart(partIndex, gjson.GetBytes(payload, "part.text").String())
		}
		// Emit done events together with output_item.done (or the next item),
		// so a late upstream done event cannot duplicate the history item.
		return repairOutcome{prefix: prefix, drop: true}, true
	}
	if kind == "response.completed" || kind == "response.incomplete" {
		prefix := make([][]byte, 0)
		updated := payload
		for i, output := range gjson.GetBytes(payload, "response.output").Array() {
			if output.Get("type").String() != "reasoning" {
				continue
			}
			prefix = append(prefix, r.openReasoning(output.Get("id").String(), int64(i))...)
			r.readReasoningItem(output)
			prefix = append(prefix, r.finishReasoning()...)
			updated, _ = sjson.SetBytes(updated, fmt.Sprintf("response.output.%d", i), r.reasoningItem())
		}
		prefix = append(prefix, r.finishReasoning()...)
		return repairOutcome{prefix: prefix, replacement: updated}, true
	}
	return repairOutcome{}, false
}

package service

import (
	"encoding/json"
	"fmt"
	"sort"
	"strings"

	"github.com/Wei-Shaw/sub2api/internal/pkg/logger"
	"github.com/google/uuid"
	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
	"go.uber.org/zap"
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
		parts := parateraReasoningContentParts(item)
		if len(parts) == 0 {
			continue
		}
		content := item.Get("content")
		if len(content.Array()) > 0 {
			// Both representations exist. The client's replay is what it just
			// persisted, so a non-empty content must not be trusted blindly: when
			// the two disagree the replay wins and the conflict is reported
			// instead of silently sending stale content upstream.
			existing := parateraReasoningContentText(item)
			if existing == "" || existing == joinParateraReasoningParts(parts) {
				continue
			}
			logger.L().Warn("paratera_deepseek.reasoning_replay_mismatch",
				zap.Int("existing_chars", len(existing)),
				zap.Int("replayed_chars", len(joinParateraReasoningParts(parts))),
			)
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

// parateraReasoningContentParts converts the reasoning summary Codex replays
// into the content parts this upstream consumes.
func parateraReasoningContentParts(item gjson.Result) []map[string]string {
	parts := make([]map[string]string, 0)
	for _, part := range item.Get("summary").Array() {
		if part.Get("type").String() != "summary_text" {
			continue
		}
		text := strings.TrimPrefix(part.Get("text").String(), deepSeekCodexReasoningHeader)
		if text != "" {
			parts = append(parts, map[string]string{"type": "output_text", "text": text})
		}
	}
	return parts
}

func joinParateraReasoningParts(parts []map[string]string) string {
	var builder strings.Builder
	for _, part := range parts {
		builder.WriteString(part["text"])
	}
	return builder.String()
}

// parateraReasoningContentText reads the text already present in the item's
// upstream-native content representation.
func parateraReasoningContentText(item gjson.Result) string {
	var builder strings.Builder
	for _, part := range item.Get("content").Array() {
		switch part.Get("type").String() {
		case "output_text", "text", "reasoning_text":
			builder.WriteString(part.Get("text").String())
		}
	}
	return builder.String()
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

// openReasoning resolves the canonical reasoning item an upstream event belongs
// to and, when it is a new item, returns the events that announce it.
//
// newItemAllowed must only be true for events that explicitly enumerate items
// (an upstream output_item.added, or the terminal response.output array).
// Summary deltas never start a second item: the upstream is known to invent a
// new id per delta, and those ids are merged onto the open item as aliases.
func (r *parateraDeepSeekItemLifecycleRepair) openReasoning(id string, index int64, newItemAllowed bool) [][]byte {
	if id != "" {
		if known, ok := r.reasoningAliases[id]; ok {
			if r.reasoning == nil || r.reasoning.done {
				r.reasoning = known
			}
			return nil
		}
	}
	if r.reasoning != nil && !r.reasoning.done {
		if id != "" {
			r.reasoningAliases[id] = r.reasoning
		}
		return nil
	}
	if r.reasoning != nil && !newItemAllowed {
		if id != "" {
			r.reasoningAliases[id] = r.reasoning
		}
		return nil
	}
	canonicalID := id
	if canonicalID == "" {
		canonicalID = "rs_" + uuid.NewString()
	}
	state := &parateraReasoningState{
		id: canonicalID, index: index,
		parts:   make(map[int64]*strings.Builder),
		started: make(map[int64]bool),
	}
	r.reasoning = state
	r.reasonings = append(r.reasonings, state)
	r.reasoningAliases[canonicalID] = state
	if id != "" {
		r.reasoningAliases[id] = state
	}
	return [][]byte{r.reasoningEvent("response.output_item.added", map[string]any{
		"item": map[string]any{"type": "reasoning", "id": canonicalID, "status": "in_progress", "summary": []any{}},
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

// text returns the accumulated reasoning exactly as it was (or will be) sent to
// the client, in summary_index order.
func (s *parateraReasoningState) text() string {
	if s == nil {
		return ""
	}
	indices := make([]int64, 0, len(s.parts))
	for index := range s.parts {
		indices = append(indices, index)
	}
	sort.Slice(indices, func(i, j int) bool { return indices[i] < indices[j] })
	var builder strings.Builder
	for _, index := range indices {
		builder.WriteString(s.parts[index].String())
	}
	return builder.String()
}

// reasoningItemText reads the text a terminal reasoning item carries, using the
// same summary/content precedence as readReasoningItem.
func reasoningItemText(item gjson.Result) string {
	parts := item.Get("summary").Array()
	if len(parts) == 0 {
		parts = item.Get("content").Array()
	}
	var builder strings.Builder
	for _, part := range parts {
		builder.WriteString(part.Get("text").String())
	}
	return builder.String()
}

// noteTerminalMismatch records a terminal snapshot that carried reasoning the
// client can no longer receive because the item was already closed. The stream
// is not reopened and nothing is silently dropped: the conflict is counted and
// logged so it is observable.
func (r *parateraDeepSeekItemLifecycleRepair) noteTerminalMismatch(itemID, terminalText string) {
	r.terminalMismatch++
	emitted := r.reasoning.text()
	logger.L().Warn("paratera_deepseek.reasoning_terminal_mismatch",
		zap.String("item_id", itemID),
		zap.Int("emitted_chars", len(emitted)),
		zap.Int("terminal_chars", len(terminalText)),
	)
}

// normalizeParateraTerminalEventType keeps the terminal event type consistent
// with the status the upstream reported, without inventing a status or an
// incomplete reason the upstream did not send.
func normalizeParateraTerminalEventType(payload []byte, kind string) []byte {
	var want string
	switch strings.TrimSpace(gjson.GetBytes(payload, "response.status").String()) {
	case "completed":
		want = "response.completed"
	case "incomplete":
		want = "response.incomplete"
	case "failed":
		want = "response.failed"
	default:
		return payload
	}
	if want == kind {
		return payload
	}
	updated, err := sjson.SetBytes(payload, "type", want)
	if err != nil {
		return payload
	}
	return updated
}

// failTerminal rewrites a terminal event to response.failed so an unrecoverable
// late-reasoning condition is explicit instead of a silent success.
func (r *parateraDeepSeekItemLifecycleRepair) failTerminal(payload []byte, kind string) []byte {
	logger.L().Warn("paratera_deepseek.reasoning_protocol_failure",
		zap.String("terminal_kind", kind),
		zap.String("reason", r.protocolFailureReason),
		zap.Int("late_reasoning_deltas", r.lateReasoningDeltas),
		zap.Int("terminal_mismatch", r.terminalMismatch),
	)
	updated, err := sjson.SetBytes(payload, "type", "response.failed")
	if err != nil {
		return payload
	}
	updated, _ = sjson.SetBytes(updated, "response.status", "failed")
	updated, _ = sjson.SetBytes(updated, "response.error", map[string]any{
		"code":    "paratera_reasoning_protocol_error",
		"message": r.protocolFailureReason,
	})
	return updated
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
		// An explicit reasoning announcement is the only event that may open a
		// second reasoning item in the same response.
		prefix := r.openReasoning(item.Get("id").String(), index, kind == "response.output_item.added")
		if kind == "response.output_item.done" {
			r.readReasoningItem(item)
			prefix = append(prefix, r.finishReasoning()...)
		}
		return repairOutcome{prefix: prefix, drop: true}, true
	}
	if strings.HasPrefix(kind, "response.reasoning_summary_") {
		prefix := r.openReasoning("", index, false)
		s := r.reasoning
		if s.done {
			// The item was already closed for the client; this delta cannot be
			// applied. Surface an explicit protocol failure instead of silently
			// dropping the content.
			r.lateReasoningDeltas++
			r.markProtocolFailure("reasoning delta arrived after the item was closed")
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
	if kind == "response.completed" || kind == "response.incomplete" || kind == "response.failed" {
		if r.protocolFailure {
			// Do not deliver a success terminal when the reasoning history could
			// not be reconciled with what the client already received.
			return repairOutcome{replacement: r.failTerminal(payload, kind)}, true
		}
		prefix := make([][]byte, 0)
		updated := payload
		// The upstream renames items between the stream and the terminal
		// snapshot (observed in every audited sample), so terminal reasoning
		// items are reconciled with the canonical items by position, not by id.
		// An id that only exists in the terminal state is registered as an alias.
		reasoningOrdinal := 0
		for i, output := range gjson.GetBytes(payload, "response.output").Array() {
			if output.Get("type").String() != "reasoning" {
				continue
			}
			if reasoningOrdinal < len(r.reasonings) {
				r.reasoning = r.reasonings[reasoningOrdinal]
				if id := strings.TrimSpace(output.Get("id").String()); id != "" {
					r.reasoningAliases[id] = r.reasoning
				}
			} else {
				// The terminal array explicitly enumerates items, so an unseen
				// reasoning item here is a real item with its own canonical state.
				prefix = append(prefix, r.openReasoning(output.Get("id").String(), int64(i), true)...)
			}
			reasoningOrdinal++
			if r.reasoning == nil {
				continue
			}
			// A snapshot can only be reconciled while the item has not been
			// closed for the client yet. Afterwards the newer content is reported
			// instead of being dropped or silently replacing delivered history.
			if terminalText := reasoningItemText(output); terminalText != "" &&
				r.reasoning.done && terminalText != r.reasoning.text() {
				r.noteTerminalMismatch(output.Get("id").String(), terminalText)
				r.markProtocolFailure("terminal reasoning differs from the delivered item")
			}
			r.readReasoningItem(output)
			prefix = append(prefix, r.finishReasoning()...)
			updated, _ = sjson.SetBytes(updated, fmt.Sprintf("response.output.%d", i), r.reasoningItem())
		}
		if r.protocolFailure {
			return repairOutcome{replacement: r.failTerminal(payload, kind)}, true
		}
		prefix = append(prefix, r.finishReasoning()...)
		updated = normalizeParateraTerminalEventType(updated, kind)
		return repairOutcome{prefix: prefix, replacement: updated}, true
	}
	return repairOutcome{}, false
}

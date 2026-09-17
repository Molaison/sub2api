package service

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"strings"

	"github.com/tidwall/gjson"
)

// Paratera's DeepSeek-V4.1-Flash Responses endpoint drops the assistant
// message item's opening announcement whenever a turn also emits reasoning or
// function_call items: the stream carries response.output_text.delta /
// .done and response.output_item.done for the message id, but never the
// matching response.output_item.added.
//
// Codex CLI keys streamed assistant text off the active item opened by
// output_item.added. Without it the delta has no owner, the client logs
// "OutputTextDelta without active item" and discards the text, so the answer
// only appears once the (complete) output_item.done arrives.
//
// The repair below rewrites the upstream stream instead of trusting it: the
// first content event that references an unannounced message item is preceded
// by the output_item.added + content_part.added pair the upstream should have
// sent. Streams that already announce the item are passed through untouched.
type parateraDeepSeekItemLifecycleRepair struct {
	reasoning *parateraReasoningState
	// announced holds item ids that were opened by an
	// response.output_item.added event, whether upstream or synthesized.
	announced map[string]struct{}
	// synthesized holds message item ids announced by this repair, so a late
	// upstream duplicate of the same announcement can be suppressed.
	synthesized map[string]struct{}
	// model is the last model name seen on the stream, echoed on synthesized
	// events to match upstream's framing.
	model string
}

func newParateraDeepSeekItemLifecycleRepair() *parateraDeepSeekItemLifecycleRepair {
	return &parateraDeepSeekItemLifecycleRepair{
		announced:   make(map[string]struct{}),
		synthesized: make(map[string]struct{}),
	}
}

// shouldRepairParateraDeepSeekStream reports whether an upstream Responses
// stream needs the message-item lifecycle repair. Scope matches the account
// level (Paratera API-key accounts running DeepSeek-V4.1-Flash), so no other
// provider's stream is rewritten.
func shouldRepairParateraDeepSeekStream(account *Account, upstreamModel string) bool {
	if account == nil || !account.IsOpenAIApiKey() {
		return false
	}
	return isParateraDeepSeekResponsesEndpoint(account.GetOpenAIBaseURL(), upstreamModel)
}

// repairOutcome describes how one upstream event must be rewritten.
type repairOutcome struct {
	// prefix holds events to emit immediately before the payload.
	prefix [][]byte
	// replacement, when non-nil, replaces the payload on the wire.
	replacement []byte
	// drop discards the payload entirely.
	drop bool
}

// Repair reports how payload must be rewritten for the client.
func (r *parateraDeepSeekItemLifecycleRepair) Repair(payload []byte) repairOutcome {
	if r == nil || len(payload) == 0 || !json.Valid(payload) {
		return repairOutcome{}
	}
	eventType := strings.TrimSpace(gjson.GetBytes(payload, "type").String())
	if eventType == "" {
		return repairOutcome{}
	}
	if model := strings.TrimSpace(gjson.GetBytes(payload, "model").String()); model != "" {
		r.model = model
	}

	if outcome, handled := r.repairReasoning(payload, eventType); handled {
		return outcome
	}
	var prefix [][]byte
	if eventType == "response.output_item.added" || eventType == "response.output_item.done" ||
		strings.HasPrefix(eventType, "response.output_text.") || strings.HasPrefix(eventType, "response.content_part.") {
		prefix = r.finishReasoning()
	}
	outcome := r.repairMessage(payload, eventType)
	outcome.prefix = append(prefix, outcome.prefix...)
	return outcome
}

func (r *parateraDeepSeekItemLifecycleRepair) repairMessage(payload []byte, eventType string) repairOutcome {
	var itemID string
	switch eventType {
	case "response.output_item.added":
		itemID = strings.TrimSpace(gjson.GetBytes(payload, "item.id").String())
		if itemID == "" {
			return repairOutcome{}
		}
		// A message announcement arriving after this repair already opened the
		// item is redundant, and forwarding it would re-open the item mid-turn.
		// The marker is kept so repeated duplicates are dropped too. Only a
		// bare announcement is dropped: one carrying content would lose data.
		if _, wasSynthesized := r.synthesized[itemID]; wasSynthesized &&
			strings.TrimSpace(gjson.GetBytes(payload, "item.type").String()) == "message" &&
			len(gjson.GetBytes(payload, "item.content").Array()) == 0 {
			return repairOutcome{drop: true}
		}
		r.announced[itemID] = struct{}{}
		return repairOutcome{}
	case "response.output_text.delta", "response.output_text.done":
		// Assistant text always belongs to the message item.
		itemID = strings.TrimSpace(gjson.GetBytes(payload, "item_id").String())
	case "response.content_part.added", "response.content_part.done":
		if strings.TrimSpace(gjson.GetBytes(payload, "part.type").String()) != "output_text" {
			return repairOutcome{}
		}
		itemID = strings.TrimSpace(gjson.GetBytes(payload, "item_id").String())
	case "response.output_item.done":
		if strings.TrimSpace(gjson.GetBytes(payload, "item.type").String()) != "message" {
			return repairOutcome{}
		}
		itemID = strings.TrimSpace(gjson.GetBytes(payload, "item.id").String())
	default:
		return repairOutcome{}
	}

	if itemID == "" {
		return repairOutcome{}
	}
	if _, ok := r.announced[itemID]; ok {
		return repairOutcome{}
	}
	r.announced[itemID] = struct{}{}
	r.synthesized[itemID] = struct{}{}
	return repairOutcome{
		prefix: r.messageItemAddedEvents(itemID, gjson.GetBytes(payload, "output_index").Int()),
	}
}

// messageItemAddedEvents builds the output_item.added + content_part.added
// pair upstream omits, mirroring the framing it emits when the items are
// announced correctly.
func (r *parateraDeepSeekItemLifecycleRepair) messageItemAddedEvents(itemID string, outputIndex int64) [][]byte {
	added := map[string]any{
		"type":         "response.output_item.added",
		"output_index": outputIndex,
		"item": map[string]any{
			"id":      itemID,
			"type":    "message",
			"role":    "assistant",
			"status":  "in_progress",
			"content": []any{},
		},
	}
	part := map[string]any{
		"type":          "response.content_part.added",
		"item_id":       itemID,
		"output_index":  outputIndex,
		"content_index": 0,
		"part": map[string]any{
			"type":        "output_text",
			"text":        "",
			"annotations": []any{},
		},
	}
	if r.model != "" {
		added["model"] = r.model
		part["model"] = r.model
	}

	events := make([][]byte, 0, 2)
	for _, event := range []map[string]any{added, part} {
		encoded, err := json.Marshal(event)
		if err != nil {
			return nil
		}
		events = append(events, encoded)
	}
	return events
}

type parateraDeepSeekStreamBody struct {
	*io.PipeReader
	source io.Closer
}

func (b *parateraDeepSeekStreamBody) Close() error {
	readerErr := b.PipeReader.Close()
	sourceErr := b.source.Close()
	if readerErr != nil {
		return readerErr
	}
	return sourceErr
}

// newParateraDeepSeekStreamBody wraps an upstream Responses SSE stream so the
// message item announcement Paratera omits is restored before the client sees
// the events that depend on it.
func newParateraDeepSeekStreamBody(source io.ReadCloser, maxLineSize int) io.ReadCloser {
	reader, writer := io.Pipe()
	body := &parateraDeepSeekStreamBody{PipeReader: reader, source: source}
	go transformParateraDeepSeekStream(source, writer, maxLineSize)
	return body
}

func transformParateraDeepSeekStream(source io.ReadCloser, destination *io.PipeWriter, maxLineSize int) {
	defer func() { _ = source.Close() }()
	if maxLineSize <= 0 {
		maxLineSize = defaultMaxLineSize
	}

	scanner := bufio.NewScanner(source)
	scanBuf := getSSEScannerBuf64K()
	defer putSSEScannerBuf64K(scanBuf)
	scanner.Buffer(scanBuf[:0], maxLineSize)
	documents := newOpenAISSEJSONDocumentScanner(scanner)
	buffered := bufio.NewWriterSize(destination, 4*1024)
	repair := newParateraDeepSeekItemLifecycleRepair()

	writeLine := func(line string) error {
		if _, err := buffered.WriteString(line); err != nil {
			return err
		}
		return buffered.WriteByte('\n')
	}
	writeFrame := func(events [][]byte) error {
		for _, event := range events {
			if err := writeLine("data: " + string(event)); err != nil {
				return err
			}
			// Synthesized frames mirror upstream's: one data field per frame.
			if err := writeLine(""); err != nil {
				return err
			}
		}
		return nil
	}
	fail := func(err error) {
		_ = buffered.Flush()
		_ = destination.CloseWithError(fmt.Errorf("repair Paratera DeepSeek stream: %w", err))
	}

	// pending holds the fields of the frame being assembled. Upstream emits one
	// "data:" field per frame, but the document scanner expands a payload that
	// packed several complete JSON documents into a single field, so a fresh
	// "data:" line flushes whatever came before it. That keeps every event in
	// its own frame no matter how the upstream (or the scanner) groups them.
	pending := make([]string, 0, 2)
	flush := func() error {
		if len(pending) == 0 {
			return nil
		}
		for _, field := range pending {
			if err := writeLine(field); err != nil {
				return err
			}
		}
		pending = pending[:0]
		return writeLine("")
	}

	for documents.Scan() {
		line := documents.Text()
		if line == "" {
			if err := flush(); err != nil {
				fail(err)
				return
			}
			continue
		}

		data, isData := extractOpenAISSEDataLine(line)
		if !isData {
			// The scanner injects "event:" fields when it splits a packed
			// payload. Every event in scope carries its type in the JSON body,
			// so those synthetic fields are dropped rather than risking a
			// field attached to the wrong frame.
			if _, isEventField := extractOpenAISSEEventLine(line); isEventField {
				continue
			}
			pending = append(pending, line)
			continue
		}

		outcome := repair.Repair([]byte(data))
		payloadLine := line
		if outcome.replacement != nil {
			payloadLine = "data: " + string(outcome.replacement)
		}
		// The repaired announcements must land before the event they unblock,
		// which means closing out any frame already being assembled.
		if len(outcome.prefix) > 0 {
			if err := flush(); err != nil {
				fail(err)
				return
			}
			if err := writeFrame(outcome.prefix); err != nil {
				fail(err)
				return
			}
		} else if len(pending) > 0 {
			if err := flush(); err != nil {
				fail(err)
				return
			}
		}
		if !outcome.drop {
			pending = append(pending, payloadLine)
		}
		// Flush semantic events promptly instead of buffering small replies
		// until the entire upstream response finishes.
		if err := flush(); err != nil {
			fail(err)
			return
		}
		if err := buffered.Flush(); err != nil {
			fail(err)
			return
		}
	}

	if err := flush(); err != nil {
		fail(err)
		return
	}

	if err := buffered.Flush(); err != nil {
		_ = destination.CloseWithError(fmt.Errorf("repair Paratera DeepSeek stream: %w", err))
		return
	}
	if err := documents.Err(); err != nil {
		_ = destination.CloseWithError(fmt.Errorf("repair Paratera DeepSeek stream: %w", err))
		return
	}
	_ = destination.Close()
}

package service

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"strings"

	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
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
	// reasonings holds every canonical reasoning item of this response in
	// first-seen order, so terminal snapshots can be matched per item instead of
	// collapsing the whole response into a single state.
	reasonings []*parateraReasoningState
	// reasoningAliases maps every upstream reasoning id seen so far (including
	// ids the upstream invents per delta) onto its canonical item.
	reasoningAliases map[string]*parateraReasoningState
	// lateReasoningDeltas counts reasoning deltas that arrived after the item was
	// already closed for the client. They cannot be applied and are counted
	// rather than silently discarded.
	lateReasoningDeltas int
	// Some upstream streams announce a message before the reasoning phase.
	// Codex has one active item, so hold empty message announcements until
	// their first text event instead of allowing reasoning to steal ownership.
	pendingMessages map[string][][]byte
	// announced holds item ids that were opened by an
	// response.output_item.added event, whether upstream or synthesized.
	announced map[string]struct{}
	// synthesized holds message item ids announced by this repair, so a late
	// upstream duplicate of the same announcement can be suppressed.
	synthesized map[string]struct{}
	// terminalMismatch counts terminal snapshots that carried reasoning content
	// the repair could no longer apply because the item was already closed and
	// sent to the client. It is surfaced (log + counter) instead of silently
	// discarding the newer content.
	terminalMismatch int
	// protocolFailure marks an unrecoverable late-reasoning case: reasoning
	// arrived after its item was already closed, or the terminal snapshot
	// carried different reasoning than what the client already persisted. The
	// terminal event is rewritten to response.failed so the caller sees an
	// explicit failure instead of a silent drop or a success with wrong history.
	protocolFailure       bool
	protocolFailureReason string
	// Canonical output_index assignment and delivered-item registry. The upstream
	// reuses/renames output_index and item ids between the stream and the terminal
	// snapshot, so indices are assigned here in delivery order and the terminal
	// response.output is rebuilt from exactly the items the client persisted.
	nextIndex      int64
	canonicalIndex map[string]int64
	deliveredItems []json.RawMessage
	doneIDs        map[string]bool
	// model is the last model name seen on the stream, echoed on synthesized
	// events to match upstream's framing.
	model string
}

// markProtocolFailure records the first unrecoverable late-reasoning condition.
func (r *parateraDeepSeekItemLifecycleRepair) markProtocolFailure(reason string) {
	if r.protocolFailure {
		return
	}
	r.protocolFailure = true
	r.protocolFailureReason = reason
}

func newParateraDeepSeekItemLifecycleRepair() *parateraDeepSeekItemLifecycleRepair {
	return &parateraDeepSeekItemLifecycleRepair{
		pendingMessages:  make(map[string][][]byte),
		announced:        make(map[string]struct{}),
		synthesized:      make(map[string]struct{}),
		reasoningAliases: make(map[string]*parateraReasoningState),
		canonicalIndex:   make(map[string]int64),
		doneIDs:          make(map[string]bool),
	}
}

// assignIndex returns the canonical output_index for an item id, allocating the
// next contiguous index on first sight.
func (r *parateraDeepSeekItemLifecycleRepair) assignIndex(id string) int64 {
	if idx, ok := r.canonicalIndex[id]; ok {
		return idx
	}
	idx := r.nextIndex
	r.nextIndex++
	r.canonicalIndex[id] = idx
	return idx
}

func setParateraIndex(payload []byte, index int64) []byte {
	updated, err := sjson.SetBytes(payload, "output_index", index)
	if err != nil {
		return payload
	}
	return updated
}

// track canonicalizes the output_index of one emitted event and records
// delivered items / rebuilds the terminal output. Every event the repair emits
// (prefix, replacement, pass-through) must go through it so the client-visible
// stream satisfies canonical index + terminal/done consistency.
func (r *parateraDeepSeekItemLifecycleRepair) track(payload []byte) []byte {
	if len(payload) == 0 || !json.Valid(payload) {
		return payload
	}
	kind := strings.TrimSpace(gjson.GetBytes(payload, "type").String())
	switch {
	case kind == "response.output_item.added":
		id := strings.TrimSpace(gjson.GetBytes(payload, "item.id").String())
		if id == "" {
			return payload
		}
		return setParateraIndex(payload, r.assignIndex(id))
	case kind == "response.output_item.done":
		id := strings.TrimSpace(gjson.GetBytes(payload, "item.id").String())
		if id == "" {
			return payload
		}
		idx := r.assignIndex(id)
		r.doneIDs[id] = true
		if item := gjson.GetBytes(payload, "item"); item.Exists() {
			r.deliveredItems = append(r.deliveredItems, []byte(item.Raw))
		}
		return setParateraIndex(payload, idx)
	case strings.HasPrefix(kind, "response.reasoning_summary_"),
		strings.HasPrefix(kind, "response.output_text."),
		strings.HasPrefix(kind, "response.content_part."),
		strings.HasPrefix(kind, "response.function_call_arguments."),
		strings.HasPrefix(kind, "response.custom_tool_call_input."):
		id := strings.TrimSpace(gjson.GetBytes(payload, "item_id").String())
		if id == "" {
			return payload
		}
		if idx, ok := r.canonicalIndex[id]; ok {
			return setParateraIndex(payload, idx)
		}
		return payload
	case kind == "response.completed" || kind == "response.incomplete" || kind == "response.failed":
		// The terminal output must be exactly what the client persisted via
		// output_item.done, in delivery order. Conflicts were already resolved in
		// planTerminal before any terminal-only item was emitted.
		output := r.deliveredAsAny()
		updated, err := sjson.SetBytes(payload, "response.output", output)
		if err != nil {
			return payload
		}
		return updated
	}
	return payload
}

// shouldRepairParateraDeepSeekStream reports whether an upstream Responses
// stream needs the message-item lifecycle repair. It gates on the model that is
// actually on the outbound body (the passthrough path does not rewrite body.model),
// so the decision always matches the request that was really sent upstream. It
// deliberately does NOT re-apply account model mapping: callers pass the final
// outbound model, and mapping it twice could select the wrong profile.
func shouldRepairParateraDeepSeekStream(account *Account, outboundModel string) bool {
	if account == nil || !account.IsOpenAIApiKey() {
		return false
	}
	return isParateraDeepSeekResponsesEndpoint(account.GetOpenAIBaseURL(), outboundModel)
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

// Repair reports how payload must be rewritten for the client. It wraps
// repairInner with a canonical-index / delivered-item pass so every emitted
// event satisfies the Responses lifecycle contract.
func (r *parateraDeepSeekItemLifecycleRepair) Repair(payload []byte) repairOutcome {
	if r == nil {
		return repairOutcome{}
	}
	out := r.repairInner(payload)

	// A duplicate output_item.done for an already-delivered item must not be
	// delivered twice. If its content changed it is a real conflict, not a
	// harmless duplicate: mark the failure and drop it so the conflicting
	// content is never handed to the client.
	if !out.drop && out.replacement == nil {
		if drop, conflict := r.duplicateDoneOutcome(payload); drop || conflict {
			if conflict {
				r.markProtocolFailure("duplicate output_item.done with different content")
			}
			for i := range out.prefix {
				out.prefix[i] = r.track(out.prefix[i])
			}
			out.drop = true
			return out
		}
	}

	effective := payload
	if out.replacement != nil {
		effective = out.replacement
	}
	// Track the prefix first so items the repair already opened (e.g. a
	// terminal-only reasoning item) are registered before the terminal snapshot
	// is matched against them.
	for i := range out.prefix {
		out.prefix[i] = r.track(out.prefix[i])
	}

	if !out.drop && effective != nil && isParateraTerminalEvent(effective) {
		// Validate/reconcile the terminal snapshot against what was actually
		// delivered BEFORE emitting anything. A conflict must fail without first
		// delivering a second, conflicting (and executable) item.
		synth, conflict := r.planTerminal(effective)
		if conflict {
			out.replacement = r.track(r.failTerminal(effective, parateraEventType(effective)))
			return out
		}
		for i := range synth {
			synth[i] = r.track(synth[i])
		}
		out.prefix = append(out.prefix, synth...)
	}
	if out.drop {
		return out
	}
	if out.replacement != nil {
		out.replacement = r.track(out.replacement)
		return out
	}
	out.replacement = r.track(payload)
	return out
}

func parateraEventType(payload []byte) string {
	return strings.TrimSpace(gjson.GetBytes(payload, "type").String())
}

func isParateraTerminalEvent(payload []byte) bool {
	switch strings.TrimSpace(gjson.GetBytes(payload, "type").String()) {
	case "response.completed", "response.incomplete", "response.failed":
		return true
	}
	return false
}

// duplicateDoneOutcome reports whether an output_item.done is a repeat of an
// already-delivered item, and whether the repeat conflicts with it. Equal content
// is dropped as a harmless duplicate; changed content is a conflict that must
// fail instead of silently discarding the newer value.
func (r *parateraDeepSeekItemLifecycleRepair) duplicateDoneOutcome(payload []byte) (drop bool, conflict bool) {
	if strings.TrimSpace(gjson.GetBytes(payload, "type").String()) != "response.output_item.done" {
		return false, false
	}
	id := strings.TrimSpace(gjson.GetBytes(payload, "item.id").String())
	if id == "" || !r.doneIDs[id] {
		return false, false
	}
	item := gjson.GetBytes(payload, "item")
	for _, raw := range r.deliveredItems {
		var d any
		if err := json.Unmarshal(raw, &d); err != nil {
			continue
		}
		if parateraItemID(d) != id {
			continue
		}
		if parateraItemContentEqual(d, item) {
			return true, false
		}
		return false, true
	}
	return true, false
}

// planTerminal one-to-one matches the terminal snapshot against the delivered
// items (by id first, then by same-type ordinal) and returns the added+done
// events for genuinely terminal-only items. A matched item whose content differs
// is a conflict: it returns conflict=true and NO synth events, so nothing extra
// is delivered before the failure.
func (r *parateraDeepSeekItemLifecycleRepair) planTerminal(payload []byte) (synth [][]byte, conflict bool) {
	terminalItems := gjson.GetBytes(payload, "response.output").Array()
	delivered := r.deliveredAsAny()
	used := make([]bool, len(delivered))

	var missing []gjson.Result
	for _, t := range terminalItems {
		tid := strings.TrimSpace(t.Get("id").String())
		ttype := strings.TrimSpace(t.Get("type").String())
		matched := -1
		for i, d := range delivered {
			if !used[i] && parateraItemID(d) == tid {
				matched = i
				break
			}
		}
		if matched < 0 {
			for i, d := range delivered {
				if !used[i] && parateraItemType(d) == ttype {
					matched = i
					break
				}
			}
		}
		if matched < 0 {
			missing = append(missing, t)
			continue
		}
		used[matched] = true
		if !parateraItemContentEqual(delivered[matched], t) {
			r.markProtocolFailure("terminal item content differs from the delivered item")
			return nil, true
		}
	}
	for _, t := range missing {
		synth = append(synth, r.terminalItemEvents(t)...)
	}
	return synth, false
}

// terminalItemEvents builds the added+done pair for a terminal-only item.
func (r *parateraDeepSeekItemLifecycleRepair) terminalItemEvents(item gjson.Result) [][]byte {
	id := strings.TrimSpace(item.Get("id").String())
	typ := strings.TrimSpace(item.Get("type").String())
	if id == "" || typ == "" {
		return nil
	}
	addedItem := map[string]any{"id": id, "type": typ, "status": "in_progress"}
	if role := strings.TrimSpace(item.Get("role").String()); role != "" {
		addedItem["role"] = role
	}
	switch typ {
	case "reasoning":
		addedItem["summary"] = []any{}
	case "message":
		addedItem["content"] = []any{}
	}
	added, err := json.Marshal(map[string]any{"type": "response.output_item.added", "item": addedItem})
	if err != nil {
		return nil
	}
	done, err := json.Marshal(map[string]any{"type": "response.output_item.done", "item": json.RawMessage(item.Raw)})
	if err != nil {
		return nil
	}
	return [][]byte{added, done}
}

// deliveredAsAny decodes the delivered canonical items in delivery order.
func (r *parateraDeepSeekItemLifecycleRepair) deliveredAsAny() []any {
	out := make([]any, 0, len(r.deliveredItems))
	for _, raw := range r.deliveredItems {
		var item any
		if err := json.Unmarshal(raw, &item); err == nil {
			out = append(out, item)
		}
	}
	return out
}

func parateraItemID(item any) string {
	if m, ok := item.(map[string]any); ok {
		if id, ok := m["id"].(string); ok {
			return id
		}
	}
	return ""
}

func parateraItemType(item any) string {
	if m, ok := item.(map[string]any); ok {
		if t, ok := m["type"].(string); ok {
			return t
		}
	}
	return ""
}

// parateraItemContentEqual compares two items ignoring the id (the upstream is
// known to rename items between the stream and the terminal snapshot).
func parateraItemContentEqual(delivered any, terminal gjson.Result) bool {
	d, ok := delivered.(map[string]any)
	if !ok {
		return false
	}
	// Compare only the semantic payload, not incidental fields (id is renamed by
	// the upstream; status/annotations may be omitted on one side).
	switch parateraItemType(d) {
	case "message":
		return parateraPartsText(d["content"]) == parateraGJSONPartsText(terminal.Get("content"))
	case "reasoning":
		dText := parateraPartsText(d["summary"])
		if dText == "" {
			dText = parateraPartsText(d["content"])
		}
		tText := parateraGJSONPartsText(terminal.Get("summary"))
		if tText == "" {
			tText = parateraGJSONPartsText(terminal.Get("content"))
		}
		return dText == tText
	case "function_call", "custom_tool_call", "tool_search_call":
		return parateraStringField(d, "name") == strings.TrimSpace(terminal.Get("name").String()) &&
			parateraStringField(d, "arguments") == strings.TrimSpace(terminal.Get("arguments").String())
	}
	return true
}

func parateraStringField(m map[string]any, key string) string {
	if s, ok := m[key].(string); ok {
		return s
	}
	return ""
}

func parateraPartsText(v any) string {
	arr, ok := v.([]any)
	if !ok {
		return ""
	}
	var b strings.Builder
	for _, p := range arr {
		if m, ok := p.(map[string]any); ok {
			if s, ok := m["text"].(string); ok {
				b.WriteString(s)
			}
		}
	}
	return b.String()
}

func parateraGJSONPartsText(r gjson.Result) string {
	var b strings.Builder
	for _, p := range r.Array() {
		b.WriteString(p.Get("text").String())
	}
	return b.String()
}

func (r *parateraDeepSeekItemLifecycleRepair) repairInner(payload []byte) repairOutcome {
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
	// A bare message announcement (message item, still without content) is held
	// back until its first text event, so it must not close the reasoning phase:
	// the upstream can keep streaming reasoning deltas after announcing it, and
	// closing here truncated the persisted reasoning to the deltas seen so far.
	if !isDeferredMessageAnnouncement(payload) &&
		(eventType == "response.output_item.added" || eventType == "response.output_item.done" ||
			strings.HasPrefix(eventType, "response.output_text.") || strings.HasPrefix(eventType, "response.content_part.")) {
		prefix = r.finishReasoning()
	}
	outcome := r.repairMessage(payload, eventType)
	outcome.prefix = append(prefix, outcome.prefix...)
	return outcome
}

// isDeferredMessageAnnouncement reports whether payload opens a message item
// that does not carry content yet. repairMessage defers such announcements, so
// Repair must defer closing the reasoning phase for them as well.
func isDeferredMessageAnnouncement(payload []byte) bool {
	if strings.TrimSpace(gjson.GetBytes(payload, "type").String()) != "response.output_item.added" {
		return false
	}
	if strings.TrimSpace(gjson.GetBytes(payload, "item.type").String()) != "message" {
		return false
	}
	return len(gjson.GetBytes(payload, "item.content").Array()) == 0
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
		if gjson.GetBytes(payload, "item.type").String() == "message" &&
			len(gjson.GetBytes(payload, "item.content").Array()) == 0 {
			if _, announced := r.announced[itemID]; !announced {
				if _, pending := r.pendingMessages[itemID]; !pending {
					r.pendingMessages[itemID] = [][]byte{append([]byte(nil), payload...)}
				}
				return repairOutcome{drop: true}
			}
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
		if eventType == "response.content_part.added" {
			if pending, ok := r.pendingMessages[itemID]; ok {
				r.pendingMessages[itemID] = append(pending, append([]byte(nil), payload...))
				return repairOutcome{drop: true}
			}
		}
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
	if pending, ok := r.pendingMessages[itemID]; ok {
		delete(r.pendingMessages, itemID)
		r.announced[itemID] = struct{}{}
		if len(pending) == 1 {
			pending = append(pending, r.messageItemAddedEvents(itemID, gjson.GetBytes(payload, "output_index").Int())[1])
		}
		return repairOutcome{prefix: pending}
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

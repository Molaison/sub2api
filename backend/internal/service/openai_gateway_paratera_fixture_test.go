package service

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/require"
	"github.com/tidwall/gjson"
)

// The fixtures under testdata/paratera_streams are real captures from the
// production gateway (DeepSeek-V4.1-Flash behind Paratera). They are replayed
// through the repair so the adapter is exercised against streams it did not
// generate, instead of only against hand-written sequences.
func TestParateraDeepSeekRepairOnCapturedStreams(t *testing.T) {
	paths, err := filepath.Glob(filepath.Join("testdata", "paratera_streams", "*.sse"))
	require.NoError(t, err)
	require.NotEmpty(t, paths, "captured fixtures must be present")

	for _, path := range paths {
		path := path
		t.Run(filepath.Base(path), func(t *testing.T) {
			raw, err := os.ReadFile(path)
			require.NoError(t, err)

			repaired := runParateraRepair(t, string(raw))
			payloads := ssePayloads(t, repaired)
			require.NotEmpty(t, payloads)

			announced := make(map[string]struct{})
			closedReasoning := make(map[string]int)
			var terminal string
			for _, payload := range payloads {
				if payload == "[DONE]" {
					continue // the SSE terminator is not an event
				}
				require.True(t, gjson.Valid(payload), "every emitted event must be valid JSON")
				switch gjsonGetString(payload, "type") {
				case "response.output_item.added":
					announced[gjsonGetString(payload, "item.id")] = struct{}{}
				case "response.output_text.delta", "response.reasoning_summary_text.delta":
					// Codex drops a delta whose item was never announced; that is
					// the defect this adapter exists to prevent.
					_, ok := announced[gjsonGetString(payload, "item_id")]
					require.True(t, ok, "delta without an owning announcement: %s",
						gjsonGetString(payload, "item_id"))
				case "response.reasoning_summary_text.done":
					closedReasoning[gjsonGetString(payload, "item_id")]++
				case "response.completed", "response.incomplete", "response.failed":
					terminal = gjsonGetString(payload, "type")
					status := gjsonGetString(payload, "response.status")
					if status != "" {
						require.Equal(t, "response."+status, terminal,
							"terminal event type must match the reported status")
					}
				}
			}
			require.NotEmpty(t, terminal, "a terminal event must survive the repair")
			for id, count := range closedReasoning {
				require.Equal(t, 1, count, "reasoning %s must be closed exactly once", id)
			}
		})
	}
}

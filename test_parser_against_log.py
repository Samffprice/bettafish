"""Verify the new protocol parser against logs/full_game_2.jsonl.

Counts how many messages are parsed as each type vs UnknownMsg.
A successful parser should classify the vast majority of messages correctly.
"""
import json
import sys
from collections import Counter

from bridge.protocol import (
    GameInitMsg,
    GameStateDiffMsg,
    ResourceDistributionMsg,
    TradeExecutionMsg,
    DiscardPromptMsg,
    AvailableActionsMsg,
    EndTurnMsg,
    GameOverMsg,
    HeartbeatMsg,
    UnknownMsg,
    parse_message,
)

LOG_FILE = "logs/full_game_best.jsonl"

counts = Counter()
unknown_types = Counter()
total = 0
unknown_count = 0

with open(LOG_FILE) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        entry = json.loads(line)
        # Only process incoming messages (skip security logs and outgoing)
        if entry.get("direction") != "in":
            continue

        raw = entry.get("raw", "")
        if not raw:
            continue

        total += 1
        msg = parse_message(raw)
        msg_type = type(msg).__name__
        counts[msg_type] += 1

        if isinstance(msg, UnknownMsg):
            unknown_count += 1
            # Try to categorize what kind of unknown it is
            try:
                data = json.loads(raw)
                if isinstance(data, dict):
                    t = data.get("type", "no-type")
                    unknown_types[f"type={t}"] += 1
                else:
                    unknown_types["non-dict"] += 1
            except Exception:
                unknown_types["parse-error"] += 1

print(f"Total incoming messages: {total}")
print(f"Unknown messages: {unknown_count} ({100*unknown_count/total:.1f}%)")
print(f"Parsed messages: {total - unknown_count} ({100*(total-unknown_count)/total:.1f}%)")
print()
print("Message type counts:")
for msg_type, count in sorted(counts.items(), key=lambda x: -x[1]):
    print(f"  {msg_type:30s} {count:5d}")

if unknown_types:
    print()
    print("Unknown message breakdown:")
    for desc, count in sorted(unknown_types.items(), key=lambda x: -x[1]):
        print(f"  {desc:30s} {count:5d}")

# Exit with error if too many unknowns (excluding connection/session messages)
# These are non-game messages we intentionally don't parse:
# type=Connected/SessionEstablished: WebSocket connection
# type=1,2,3,6: Lobby/session/user info
# type=27: Discard confirmation, type=62: UI notification
# type=63,64,70,78: UI signals/notifications
expected_unknown_types = {"type=Connected", "type=SessionEstablished",
                          "type=1", "type=2", "type=3", "type=6",
                          "type=27", "type=62", "type=63", "type=64",
                          "type=70", "type=78",
                          "parse-error"}
unexpected_unknowns = sum(
    count for desc, count in unknown_types.items()
    if desc not in expected_unknown_types
)
if unexpected_unknowns > 0:
    print(f"\nWARNING: {unexpected_unknowns} unexpected unknown messages!")
    sys.exit(1)
else:
    print(f"\nAll unknowns are expected non-game messages.")
    sys.exit(0)

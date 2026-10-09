
from meg_tokens.meg.meg_bids import _combined_event_id


def test_combined_event_id_respects_subject_overrides():
    # H06 has Start/Go codes swapped relative to every other subject (see
    # meg_tokens.meg.epoching.SUBJECT_EVENT_OVERRIDES).
    event_id = _combined_event_id("H06")
    assert event_id["Go"] == 262144
    assert event_id["Start"] == 524288

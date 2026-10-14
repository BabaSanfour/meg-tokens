
from meg_tokens.meg.meg_bids import trigger_codes


def test_trigger_codes_respect_subject_overrides():
    # H06 has Start/Go codes swapped relative to every other subject.
    event_id = trigger_codes("H06")
    assert event_id["Go"] == 262144
    assert event_id["Start"] == 524288

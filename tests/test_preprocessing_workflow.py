import mne
import numpy as np
import pandas as pd

from meg_tokens.core import EpochingConfig, ProjectConfig
from meg_tokens.io import DerivativeLayout, save_table
from meg_tokens.workflows import epoch_subjects


def test_epoch_subjects_consumes_staged_raw_and_behavior(tmp_path):
    layout = DerivativeLayout(tmp_path)
    raw_path = layout.preprocessed_raw(
        subject="H1",
        run="1",
        condition="Slow",
        processing="filt",
    )
    raw_path.parent.mkdir(parents=True)
    info = mne.create_info(
        ["MEG001", "STI 014"],
        sfreq=1000.0,
        ch_types=["mag", "stim"],
    )
    data = np.zeros((2, 3000))
    data[1, 1000] = 524288
    data[1, 2000] = 524288
    mne.io.RawArray(data, info).save(raw_path, overwrite=True)

    behavior_path = layout.behavior(subject="H1", run="1", condition="Slow")
    behavior = pd.DataFrame(
        {
            "subject": ["H01", "H01"],
            "condition": ["Slow", "Slow"],
            "run": [1, 1],
            "source_file": ["H1Slow1_recording.tdms"] * 2,
            "nTrialIndex": [1, 2],
            "sTrialClass": [1, 2],
            "nInitialTime": [0, 0],
            "nChoiceMade": [1, 2],
            "nCorrectChoice": [1, 2],
            "tGO": [1000, 2000],
            "tEnterTarget": [1400, 2400],
            "tTrialEnd": [1600, 2600],
            "sTokenDirs": ["121", "212"],
            "tTime": ["[1100]", "[2100]"],
            "nProb": ["[0.6]", "[0.6]"],
            "rawRT": [400, 400],
            "isCorrect": [True, True],
        }
    )
    save_table(behavior_path, behavior, metadata={"stage": "behavior_parsing"})

    result = epoch_subjects(
        ProjectConfig(bids_root=tmp_path),
        subjects=["H1"],
        settings=EpochingConfig(tmin=-0.1, tmax=0.2, alignment="go"),
    )

    assert result.stage == "meg_epoching"
    assert layout.epochs(
        subject="H1",
        run="1",
        condition="Slow",
        alignment="go",
    ) in result.outputs
    epochs = mne.read_epochs(
        layout.epochs(
            subject="H1",
            run="1",
            condition="Slow",
            alignment="go",
        ),
        preload=False,
    )
    assert len(epochs) == 2
    assert epochs.metadata["nTrialIndex"].tolist() == [1, 2]

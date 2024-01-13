# HMS - Harmful Brain Activity Classification
## Overview
> The goal of this competition is to detect and classify seizures and other types of harmful brain activity. You will develop a model trained on electroencephalography (EEG) signals recorded from critically ill hospital patients.
>
> Your work may help rapidly improve electroencephalography pattern classification accuracy, unlocking transformative benefits for neurocritical care, epilepsy, and drug development. Advancement in this area may allow doctors and brain researchers to detect seizures or other brain damage to provide faster and more accurate treatments.

There are six patterns of interest for this competition: seizure (SZ), generalized periodic discharges (GPD), lateralized periodic discharges (LPD), lateralized rhythmic delta activity (LRDA), generalized rhythmic delta activity (GRDA), or “other”

The EEG segments are given three types of labels.
- idealized patterns: high levels of agreement of experts
- proto patterns: ~1/2 of experts give a label as “other” and ~1/2 give one of the remaining five labels
- edge cases: experts are approximately split between 2 of the 5 named patterns

## Summary
### Data
Two sequence data
- Electroencephalography (EEG) signals recorded from critically ill hospital patients
- Spectrograms assembled EEG data

Annotated data
> - `eeg_id` - A unique identifier for the entire EEG recording.
> - `eeg_sub_id` - An ID for the specific 50 second long subsample this row's labels apply to.
> - `eeg_label_offset_seconds` - The time between the beginning of the consolidated EEG and this subsample.
> - `spectrogram_id` - A unique identifier for the entire EEG recording.
> - `spectrogram_sub_id` - An ID for the specific 10 minute subsample this row's labels apply to.
> - `spectogram_label_offset_seconds` - The time between the beginning of the consolidated spectrogram and this subsample.
> - `label_id` - An ID for this set of labels.
> - `patient_id` - An ID for the patient who donated the data.
> - `expert_consensus` - The consensus annotator label. Provided for convenience only.
> - `[seizure/lpd/gpd/lrda/grda/other]_vote` - The count of annotator votes for a given brain activity class. The full names of the activity classes are as follows: lpd: lateralized periodic discharges, gpd: generalized periodic discharges, lrd: lateralized rhythmic delta activity, and grda: generalized rhythmic delta activity . A detailed explanations of these patterns is available here.

### Evaluation
> Submissions are evaluated on the Kullback Liebler divergence between the predicted probability and the observed target.
> For each eeg_id in the test set, you must predict a probability for each of the vote columns. The file should contain a header and have the following format:
> 
> eeg_id,seizure_vote,lpd_vote,gpd_vote,lrda_vote,grda_vote,other_vote
> 0,0.166,0.166,0.167,0.167,0.167,0.167
> 1,0.166,0.166,0.167,0.167,0.167,0.167
> etc.
>
> Your total predicted probabilities for each row must sum to one or your submission will fail.

## References
- [American Clinical Neurophysiology Society’s Standardized
Critical Care EEG Terminology: 2021 Version](https://www.acns.org/UserFiles/file/ACNSStandardizedCriticalCareEEGTerminology_rev2021.pdf)
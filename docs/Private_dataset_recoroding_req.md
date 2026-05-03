Here is exactly what you need from the NYU private dataset:

---

**Hardware requirements:**
- Minimum 32-channel EEG system (64-channel preferred)
- Must include electrodes: FC3, FCZ, FC4, CP3, CPZ, CP4 (for 11-channel upgrade)
- Standard 10-20 or 10-10 electrode placement
- Sampling rate: minimum 500Hz (1000Hz or 2048Hz preferred)
- Output format: EDF, BDF, or GDF (all readable by MNE)
- Dedicated EOG channels if available (helps artifact removal)

---

**Subjects:**
- Minimum 3 subjects (4-5 preferred)
- Right-handed (same as PhysioNet and BCI IV-2a protocol)
- No neurological conditions
- No prior BCI experience preferred (naive subjects generalize better)
- Age 18-35 (consistent with other datasets)

---

**Trial structure (must match BCI IV-2a protocol exactly):**
- t=0s: fixation cross appears with beep
- t=2s: directional arrow cue appears (left or right)
- t=5s: fixation cross disappears
- t=5-6s: rest period
- t=6s: next trial begins
- Epoch we extract: -1s to +3s relative to cue onset

---

**Trial counts:**
- Minimum 50 trials per class per subject
- Classes: left hand imagery, right hand imagery, rest
- Total minimum: 150 trials per subject
- Total minimum across all subjects: 450 trials
- Preferred: 100 trials per class per subject

---

**Annotations and events:**
- Clear digital trigger markers for each cue onset
- Left cue marker clearly distinguishable from right cue marker
- Rest period markers if possible
- Artifact markers for any known bad segments
- All markers saved in the EEG file not a separate file

---

**Recording conditions:**
- Subject seated comfortably
- Screen at eye level showing fixation cross and arrow cues
- Room as quiet as possible to minimize EMG artifacts
- Consistent impedance below 10 kOhm for all electrodes
- Minimum 5 minute break between runs to reduce fatigue

---

**Session structure:**
- Minimum 3 runs per subject
- Each run: 50 trials (mix of left and right)
- Rest between runs: 5 minutes minimum
- Practice block before recording: 10 trials not included in data

---

**Metadata to record and save:**
- Subject ID (anonymized, e.g. NYU01, NYU02)
- Age and handedness
- EEG system make and model
- Electrode cap size used
- Sampling rate confirmed
- Date and time of recording
- Any notes about artifacts or subject behavior during recording
- Impedance values before recording starts

---

**What to avoid:**
- Do not use the same subjects as any public dataset
- Do not let subjects practice BCI before recording naive baseline
- Do not record immediately after physical exercise (EMG artifacts)
- Do not use wireless systems with known packet loss
- Do not use consumer-grade EEG (Emotiv, Muse) — not research grade

---

**Nice to have but not essential:**
- Video recording of sessions for artifact verification
- EMG channels on forearm to verify no actual movement
- Eye tracking to cross-validate EOG artifact removal
- Multiple sessions per subject (session 1 and session 2) for cross-session evaluation
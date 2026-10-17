# Band power and event-related fields: hypotheses to test

What the literature and the tokens task predict for the two derivatives the
MEG pipeline now holds, band power and event-related fields, in the HCP-MMP1
regions, the aseg deep structures and on fsaverage. Each hypothesis gives the
claim and its source, the test on our data, the data it reads, the controls
it needs and the figure. Three literature passes (band power, event-related
fields, the task and its models) were merged here; where they disagreed the
disagreement is kept as competing predictions. Nothing here has been run
except the motor-beta exploration it starts from (§1).

Notation. **SP(t)** is the success probability of the right target after each
token (in the trial table); **E(t) = SP(t) − 0.5** is signed evidence.
**U(t)** is motor beta averaged over hemispheres in the motor region (HCP 4,
3a, 3b, 1, 6d), sign-flipped so that desynchronisation is positive.
**Lat(t)** is lateralisation, right minus left hemisphere, so positive means
preparing the right hand (side 1). "Contra" is the hemisphere opposite the
chosen hand. Region names drop the `L_`/`R_` prefix and `_ROI` suffix. Bands:
theta 5–8, alpha 9–13, beta 16–24, gamma 30–60, high gamma 60–90 Hz.

## Summary

Level: **R** region-wise (HCP-MMP1 / aseg), **V** vertex-wise on fsaverage,
**S** sensors. Data: **cm** condition means (12), **qm** decision-time
quintile means (10), **st** single-trial band power, **erp** condition ERFs,
**fs** fsaverage, **Dn** a derivative still to build (§6). Strength is how
likely the effect is to appear; novelty how new it would be for this task in
humans.

| ID | Hypothesis | Lock | Measure | Level | Data | Strength | Novelty |
|---|---|---|---|---|---|---|---|
| **A. Controls that come first** |||||||
| A1 | Press: contralateral motor field, MEF-I and high-gamma burst; sign anchor | enter | ERF, HG | R, S, V | erp, cm | ★★★★★ | low |
| A2 | Each token evokes an M100 that adapts over tokens | go | ERF | R, V | erp | ★★★★★ | low |
| A3 | Token harmonics do not make the beta ramp (induced vs total power) | go, decision | beta | R | st, erp | control | — |
| A4 | Post-press token rate: establish it, then check its tag | enter | spectra | R | D0 | control | — |
| A5 | Ocular and muscle sentinels: frontal/temporal lateralisation, nine-subject split | all | all | R, S | all | control | — |
| **B. Urgency** |||||||
| B1 | U(t) is a clock: elapsed time, not evidence | go | beta | R | st | ★★★★ | med |
| B2 | Level at commitment: fixed contra threshold vs constant product vs capped parietal signal | decision | beta, ERF | R | st, qm, D5 | ★★★★ | high |
| B3 | Ambiguous trials commit on time, at higher U | go, decision | beta | R | st | ★★★ | med |
| B4 | A medial readiness field shares the ramp | go, decision | ERF < 4 Hz | R | D4 | ★★ | high |
| B5 | Each subject's ramp slope tracks their speed and fitted urgency | go | beta | R | st | ★★ | med |
| **C. Evidence × urgency** |||||||
| C1 | Lat's gain on evidence scales with U(t) (multiplication) | go | beta, alpha | R | st | ★★★★ | high |
| C2 | Evidence memory is short (τ ≈ 0.1–0.3 s) | go | beta, alpha | R | st | ★★★ | high |
| C3 | Misleading trials: wrong-hand Lat, reversal within τ of the evidence crossing | go, crossing | beta, LRF | R, S | st, erp | ★★★★ | high |
| C4 | Lateralised readiness field before commitment | decision | ERF | S, R | erp | ★★★★ | med |
| C5 | Frontoparietal evidence encoding with an IPS → PMd → M1 latency gradient | token | alpha, beta | R → V | st, fs | ★★★ | high |
| C6 | Neural kernel: primacy vs recency within decision-time strata | decision | beta | R | st | ★★★ | high |
| C7 | A parietal/PCC build-up (CPP analogue) ramps to commitment | decision | ERF | R, V | erp, D4 | ★★★ | med |
| **D. Token-level encoding** |||||||
| D1 | Token direction lateralises occipito-parietal responses | go, token | ERF, gamma | R, S, V | erp, st, fs | ★★★ | med |
| D2 | The late token response scales with \|ΔSP\| (decision relevance) | token | ERF, alpha, theta | R, V | D1, st | ★★★ | high |
| D3 | Reversal tokens: urgency gating vs suppression vs change-point boost; midfrontal theta | token | ERF, theta | R | D1, st | ★★ | high |
| D4 | Token gain rises with elapsed time and in Fast | go | ERF, 5 Hz | R, S | erp, D6 | ★★★ | high |
| D5 | The 5 Hz tag spreads from V1 to IPS and PMd with a latency gradient | go | 5 Hz SNR | V | fs | ★★★★ | med |
| D6 | Commitment is phase-locked to token arrivals | decision | 5 Hz | R, S | erp | ? | high |
| **E. Commitment and after** |||||||
| E1 | After commitment, motor regions stop encoding tokens; visual cortex does not | decision, enter | beta, gamma, ERF | R | st, D1 | ★★★★ | high |
| E2 | Neural change-point 250–450 ms before the press; alpha bends before beta | enter | alpha, beta | R | qm, st | ★★★ | med |
| E3 | Posterior alpha suppressed while deliberating, rebounds after | decision, enter | alpha | R | cm, qm | ★★★★ | low |
| E4 | Post-decision tokens stay encoded; a Pe analogue locks to the decisive token | enter, token | ERF, theta | R | D0, D1, D2 | ★★ | high |
| E5 | The post-movement beta rebound scales with success probability | enter | beta | R | st | ★★★ | med |
| **F. Fast vs Slow** |||||||
| F1 | The block effect is an offset present before the trial | go | beta, alpha, ERF | R → V | st, fs, D4 | ★★★ | high |
| F2 | The ×1.12 stretch sits in evidence gain, not in the urgency rate | go | beta | R | st | ★★★ | high |
| **G. Monitoring, outcome, history** |||||||
| G1 | Medial-frontal response at the press tracks low SP at commitment, not correctness | enter | ERF, theta | R | erp, D2 | ★★★ | med |
| G2 | The outcome signal moves from feedback to the decisive token | feedback, token | ERF, theta, beta | R | D1, D3 | ★★ | high |
| G3 | After an error: higher pre-trial beta, lower posterior alpha, flatter ramp | go | beta, alpha, theta | R | st | ★★★ | med |
| G4 | Residual beta from the previous press drives choice alternation | go (pre) | beta | R | st | ★★★ | high |
| G5 | Pre-trial posterior alpha predicts decision time | go (pre) | alpha | R | st | ★★ | low |
| **H. Exploratory and confounded** |||||||
| H1 | Posterior alpha lateralisation follows the leading target | decision | alpha | R | st | ★★ | med |
| H2 | Choice-predictive motor gamma before commitment | decision | gamma, HG | R | st | ★ | med |
| H3 | Parietal/frontal gamma tracks evidence | decision | gamma | V | fs | ★ | med |
| **I. Deep structures** |||||||
| I1 | Pallidal/striatal urgency against a forward-simulated leakage null | go | beta, ERF | R (aseg) | st, erp | ★ | med |
| I2 | The preprint's subcortical speed–accuracy effect is neck muscle | go | gamma, HG | R (aseg) | st | control | high |
| I3 | Cerebellar ramp ipsilateral to the hand | go, enter | beta | R (aseg) | st | ★★ | med |

## 1. Starting point

From the condition and quintile means, motor-region beta (4, 3a, 3b, 1, 6d,
both hemispheres) descends on one ramp from the first token, at the same
rate for Fast and Slow and for every decision-time quintile. Each decision
commits somewhere on it: beta at commitment falls with log decision time
(p = 3.5e-19 DSS, 1.9e-21 ICA); every group reaches the same trough at the
press (p = .999). Choice lateralisation builds before commitment, earlier for
easy than misleading trials, which the decision-time difference partly
explains. B2, C1–C3 and F1–F2 are the hypotheses this finding sharpens.

## 2. Facts that constrain every test

Checked against our own docs and code.

1. **The decision lock is the press shifted by a constant** (press minus the
   hand-specific motor latency from the RT runs, per subject). The two locks
   hold the same waveforms; "sharper at commitment than at the press" is not
   evidence. Only transition latencies (E2) or a commitment time estimated
   per trial from a model can separate them.
2. **The data are high-passed at 0.5 Hz** (`preprocessing.HIGH_PASS`). Ramps
   longer than about 1 s are attenuated and can invert (Tanner et al. 2015,
   https://escholarship.org/uc/item/1p7352t2). Slow evoked ramps (B4, C7, F1
   as an ERF) are lower bounds until a re-epoching with a lower high-pass (D4).
3. **The token stream sits inside the bands.** 5 Hz tokens put their
   fundamental in theta, the 10 Hz harmonic in alpha and the 15/20 Hz
   harmonics in beta. Every band-power result has to survive on induced
   power (A3).
4. **The post-press token rate is not established for this dataset.** In the
   monkey task the remaining tokens jump every 50 ms (Fast) or 150 ms (Slow),
   which would put a 20 Hz flicker in beta for Fast and 6.7 Hz in theta for
   Slow. `behavior.md` records that the remaining-token timing schedule was
   never established from the acquisition material. Until it is (D0), every
   post-press Fast-vs-Slow contrast is suspect, and post-press analyses run
   within one speed.
5. **Hand, target and hemifield are one variable.** Side 1 is the right hand
   and the right target. Motor lateralisation is interpretable; occipital and
   parietal lateralisation cannot be attributed to the hand or to space.
6. **Token improbability is constant.** In random trials each token goes
   either way with probability 0.5 (Cisek et al. 2009,
   https://doi.org/10.1523/JNEUROSCI.1844-09.2009); the classes are built
   sequences. What varies from token to token is decision relevance, |ΔSP| or
   the posterior KL, not surprise.
7. **The feedback lock exists in sensor epochs** (16,290 epochs) but has no
   band-power or ERF derivative; the press and feedback carry no trigger, so
   both sit at their behavioural offset.
8. **Deep structures are 77–93 % cortex** through this inverse (cerebellum
   about 50 %, and next to neck muscle); 360 regions exceed the data's rank
   (about 250), so neighbouring regions are one effect.
9. **Nine subjects carry tonic neck muscle**; saccades reach orbitofrontal,
   insular and temporal-pole sources; DSS removes ocular components that ICA
   keeps, so a difference between streams is itself diagnostic.
10. **Zero-phase filters carry later activity backwards** by half their
    length (beta about 0.22 s, theta about 0.8 s). A claim that something
    happens *before* an event has to clear that margin.

## 3. Shared conventions

**Normalisation.** Log10 power centred per subject × region × band × stream
(no pre-trial baseline, which would erase block offsets in F1). A common
pre-go baseline pooled over conditions is for display only.

**Evoked sign.** Region time courses are `mean_flip`, so a region's sign is a
convention of each subject's anatomy. Four ways round it, named in the tests:
- **S1, orthogonal alignment.** Per subject and region, flip so the mean over
  all twelve conditions is positive in the window; condition contrasts are
  unbiased by it.
- **S2, anchor alignment.** Take the sign from an independent landmark: the
  press-locked motor field in 4/3b (A1) or the first-token peak in V1 (A2).
- **S3, sign-free.** RMS or |x|; inflated in small cells by about 1/√nave, so
  equalise trial counts or subtract a noise floor (D8).
- **S4, vertex-wise.** The fsaverage normal component follows one convention
  in every subject.

**Token re-indexing.** Token k appears at go + 200·(k − 1) ms; in the decision
lock at 200·(k − 1) − t_commit. Token-locked windows (−0.1..+0.4 s) can be cut
from the stored single-trial band power by index arithmetic. The go lock
covers tokens 1–5; later tokens come from the decision lock, masking samples
before the first token on short decisions. Token-locked *evoked* responses
need single-trial source ERFs (D1).

**Statistics.** The subject is the unit (32 subjects, all 319 recordings).
- Time courses: per-subject regression or contrast, then
  `mne.stats.permutation_cluster_1samp_test` over time.
- Vertex maps: `spatio_temporal_cluster_1samp_test` with
  `mne.spatial_src_adjacency` on fsaverage oct6.
- Scalars: linear mixed models with random intercepts and slopes per subject
  and a run term.
- Fast/Slow: permute run labels within subject (8 runs, 70 relabellings;
  coco-pipe `blocks_within_strata`).
- Decoding through coco-pipe with within-subject or block permutation.
- Nulls with equivalence tests (TOST). Regions frozen before testing.

**Every result passes:**

| | Check | How |
|---|---|---|
| C1 | Decision time | Match on decision-time quantiles within subject × condition; elapsed time as a covariate; go-locked bins keep only trials still deliberating |
| C2 | Quintile composition | Quintiles within class, or reweighted to one class mix; define on half the trials, measure on the other (regression to the mean) |
| C3 | Streams | Same sign and a significant cluster in DSS and ICA |
| C4 | Leakage | `make_inverse_resolution_matrix`, `get_cross_talk`, `resolution_metrics`; forward-simulation null (`SourceSimulator`, `apply_forward`); LCMV replication; laterality as a signature leakage cannot make |
| C5 | Muscle | Nine tonic-muscle subjects vs the other 23 for any gamma, high-gamma or deep effect; per-run 110–140 Hz power as a covariate |
| C6 | Eyes | Frontal/temporal sentinels (A5); EOG lateralisation against E; DSS vs ICA |
| C7 | Token stream | Induced power (`Epochs.subtract_evoked`), ITC (`tfr_array_morlet`, `output="itc"`), bins on the 200 ms grid |
| C8 | Commitment error | At least the RT-run SD; repeat on the press lock; shift ±50 ms |
| C9 | Filter smearing | Clear half the filter length, or redo with `create_filter(..., phase="minimum")`; simulate any timing claim through the same filter and Hilbert |
| C10 | Session | Trial index, run, block order, head position per run; Fast has more trials (281 vs 229 per subject) |

## 4. The hypotheses

### A. Controls that come first

**A1. The press: motor field, MEF-I and high-gamma burst.** A press produces
a contralateral motor field just before the movement, a reafferent MEF-I
about 100 ms after it (Cheyne et al. 2006,
https://pmc.ncbi.nlm.nih.gov/articles/PMC6871358) and a brief contralateral
60–90 Hz burst in M1 (Cheyne et al. 2008, https://orca.cardiff.ac.uk/id/eprint/32384/;
Cheyne & Ferrari 2013, https://pmc.ncbi.nlm.nih.gov/articles/PMC3774986/).
- *Test.* Enter lock, 4, 3a, 3b, 1, 2, contra vs ipsi from right- vs
  left-choice conditions; sensors for topographies. Cluster test −0.5..+0.5 s
  on hand × speed × class; MEF-I equivalent across conditions (TOST). High
  gamma: contralateral peak 0–100 ms after the press. The burst's latency
  against the beta trough checks the motor-latency estimate.
- *Use.* Sets each subject's sign for 4/3b (S2) and validates the press time.
- *Controls.* C5 (muscle), C3.
- *Figure.* Left- and right-press topographies; 4/3b traces by speed;
  time × region high-gamma lateralisation strip; fsaverage map at the peak.

**A2. Every token evokes an M100 that adapts.** Each jump evokes an occipital
response at 100–130 ms, largest for the first token, with repetition
suppression after (Norcia et al. 2015,
https://pmc.ncbi.nlm.nih.gov/articles/PMC4581566).
- *Test.* Go lock, S3 in V1, V2, V3, V4, V3A, MT, MST, LO1–3. Peak 80–160 ms
  after each jump at 0, 200, …, 800 ms against −50..0 ms before it. Mixed model
  amplitude ~ token × speed + class. Vertex-wise cluster test on the first
  token (S4).
- *Use.* V1 sign anchor (S2); confirms the timing of the stimulus channel.
- *Figure.* RMS with token ticks; fsaverage map at the peak; amplitude by token.

**A3. Token harmonics do not make the beta ramp.** The phase-locked token
response has 15/20 Hz harmonics that could reach M1/PMd and imitate the ramp.
**Critical for §1.**
- *Test.* Areas 4/6d: evoked 15/20 Hz amplitude over time (go and decision
  locks); recompute the ramp on induced power (condition ERF subtracted per
  run before the Hilbert, or `Epochs.subtract_evoked`). Expected: small, flat
  evoked harmonics; the ramp unchanged.
- *Data.* ERFs plus a re-run of band power on induced data for the motor
  regions (cheap: a handful of regions, single lock).
- *Figure.* Ramp from total vs induced power; evoked 20 Hz amplitude.

**A4. The post-press token rate and its tag.** First establish the rate from
the acquisition material or the photodiode/stimulus channel (D0). Then
narrow-band spectra from the press to the trial's end
(`psd_array_multitaper`) in V1–V4 and the motor regions, by condition, with
ITC at the tagging frequency. Any post-press Fast/Slow beta difference with
the tag's spatial profile is visual, not motor.

**A5. Ocular and muscle sentinels.** The saccadic spike field is lateral and
strongest at frontal and temporal sensors (Carl et al. 2012,
https://neuinfo.org/literature/pmid:21963912); muscle is broadband above
30 Hz (Muthukumaraswamy 2013, https://doi.org/10.3389/fnhum.2013.00138).
Repeat every lateralised contrast (C3, C4, D1, H1) in MLF/MRF/MLT/MRT
sensors, orbitofrontal and TGd regions and the EOG averages (D7), DSS vs ICA.
Effects found only there, or in one stream, are ocular. Every gamma or
high-gamma effect is shown without the nine tonic-muscle subjects.

### B. Urgency

**B1. U(t) is a clock, not an evidence signal.** Urgency grows with time and
ignores evidence (Murphy et al. 2016,
https://pmc.ncbi.nlm.nih.gov/articles/PMC5123079/; Thura & Cisek 2016,
https://pmc.ncbi.nlm.nih.gov/articles/PMC6602002/; in the tokens task, Thura
& Cisek 2014, https://doi.org/10.1016/j.neuron.2014.01.031).
- *Test.* Go lock, 200 ms bins on the token grid, each cut at the trial's
  commitment. U in the motor region and in SCEF, 6ma, 6mp. Mixed model
  U ~ t + |E| + Σ|E| + class + condition. Expected: t dominates, evidence
  terms ≈ 0 (TOST), the same ramp in easy, ambiguous and misleading trials in
  token time. Integration without urgency: U follows Σ|E|, steeper for easy.
- *Controls.* t and token count are collinear, so the leverage is |E| and
  misleading vs ambiguous at matched t; survivor bias (C1); A3.
- *Figure.* U(t) by class; each coefficient's time course with its CI.

**B2. The level at commitment: three competing accounts.** Beta at
commitment is lower for longer decisions (§1). Three accounts predict this,
and they disagree on the parts:
- **Fixed contralateral threshold.** Contralateral beta reaches a fixed level
  at commitment; urgency lowers beta in both hemispheres, and the
  dt-dependence lives in the ipsilateral hemisphere (O'Connell et al. 2012,
  https://wwwha.tcd.ie/Neuroscience/assets/PDF/OConnell_Dockree_Kelly_NNpublished.pdf;
  Kelly & O'Connell 2013, https://pmc.ncbi.nlm.nih.gov/articles/PMC6618757/;
  Steinemann et al. 2018, https://pmc.ncbi.nlm.nih.gov/articles/PMC6128824/;
  Kelly et al. 2021, https://doi.org/10.1038/s41562-020-00967-9).
  Prediction: contra slope on dt ≈ 0, ipsi slope negative, so |Lat| at
  commitment shrinks with dt.
- **Constant product (urgency gating).** Filtered E × U reaches a fixed bound
  (Thura et al. 2012, https://doi.org/10.1152/jn.01071.2011); later
  decisions commit with less evidence and more urgency (SP at decision 0.78
  easy vs 0.65 hard). Prediction: across quintiles and classes,
  log|Lat_c| = a − k·log U_c with k = 1; a fixed evidence bound gives k = 0.
- **Capped parietal signal.** Urgency lowers the level the evidence signal
  (CPP) reaches at the response while the motor level stays fixed
  (Steinemann et al. 2018; Kelly et al. 2021). Prediction: the parietal/PCC
  build-up at commitment is lower for late decisions and in Fast; motor
  beta at the press is not.
- *Test.* Decision lock, −100..0 ms. Per trial: contra and ipsi log beta in
  the motor region, mixed model beta ~ hemisphere × dt + condition + class;
  then |Lat_c| and U_c per subject × condition × class × quintile for the
  k fit, quintiles defined on half the trials (C2). Parietal: ERF amplitude in
  the C7 regions by quintile (D5).
- *Figure.* Contra and ipsi beta at commitment against quintile, over the
  shared ramp; log|Lat_c| against log U_c with k = 0 and k = 1 lines;
  parietal level and motor level against quintile.

**B3. Ambiguous trials commit on time.** With evidence near zero, the
urgency-gating model commits through rising U (Thura et al. 2012).
- *Test.* Ambiguous vs easy within a quintile: same U slope in token time,
  higher U at commitment, Lat small until late.
- *Figure.* U(t) and Lat(t), ambiguous vs easy, commitment marked.

**B4. A medial readiness field shares the ramp.** The readiness potential
behaves as accumulation to a bound driven by urgency-like input (Schurger et
al. 2012, https://doi.org/10.1073/pnas.1210467109); its early phase is
medial (Shibasaki & Hallett 2006,
https://doi.org/10.1016/j.clinph.2006.04.025); the CNV tracks caution (Boehm
et al. 2014, https://doi.org/10.1016/j.neuroimage.2014.03.050).
- *Test.* 6ma, 6mp, SCEF, 24dd, 24dv, p24pr, 4 (S1), low-passed below 4 Hz to
  remove the token stream. Go-lock slope 0.2–1.0 s by speed × class; level at
  −100..0 ms of the decision lock, predicted equal across conditions;
  across-subject correlation with the beta slope.
- *Data.* D4. On the current 0.5 Hz data only a qualitative check.
- *Figure.* Medial slow field against time from the first token, over U(t).

**B5. Each subject's ramp tracks their speed.** Behaviour has one speed axis
(mean decision time against urgency slope, r = −0.53); urgency as a trait
(Carland et al. 2019, https://doi.org/10.1177/1073858419841553).
- *Test.* Spearman of each subject's U slope with fitted urgency and mean
  decision time; the Fast − Slow neural offset with the behavioural
  adjustment. n = 32 detects |r| ≥ 0.47 at 80 % power; report split-half
  reliability.

### C. Evidence × urgency

**C1. Lat's gain on evidence scales with U(t).** The decision variable is
low-pass-filtered evidence × urgency (Cisek et al. 2009; Thura et al. 2012);
an EEG CPP followed evidence × urgency in a related task (Yau et al. 2021,
https://doi.org/10.1101/2020.10.02.323683). Lateralised sensorimotor beta
tracks the evidence (Donner et al. 2009, https://doi.org/10.1016/j.cub.2009.07.066;
de Lange et al. 2013, https://doi.org/10.1523/JNEUROSCI.4227-12.2013).
- *Test.* Single trials at each pre-commitment token: Lat ~ E + t + E×t +
  ∫E. Model-free: the slope of Lat on E in each 200 ms bin, correlated with
  the subject's own U(t) across bins and quintiles; predicted slope(t) ∝ U(t).
  Lat is fixed as right minus left, not relative to the choice, to avoid
  circularity.
- *Controls.* When evidence is stable ∫E ≈ E·t and the models mimic each
  other (Hawkins et al. 2015, https://pmc.ncbi.nlm.nih.gov/articles/PMC4495756/);
  report VIFs and fit on trials where E changes (misleading, ambiguous with
  excursions). Standardise E within token index. A3.
- *Figure.* Slope of Lat on E against time, over scaled U(t), by condition.

**C2. Evidence memory is short.** The urgency-gating filter has τ of
100–250 ms (the behavioural fits here use 200 ms and hold at 100 and 300);
PMd follows the current SP (Thura & Cisek 2014); human MEG motor beta
follows normative, imperfect accumulation (Murphy et al. 2021,
https://doi.org/10.1101/2020.01.29.924795).
- *Test.* Per subject, Lat(t) = a + b·U(t)·[h_τ * E](t), h_τ exponential, τ on
  a grid from 50 ms to 5 s plus ∞ (perfect integration). Pick τ by
  leave-one-run-out R². Alternative input: the token difference N_R − N_L,
  itself an integral.
- *Resolution.* The beta filter smears Lat by about 200 ms; τ below about
  150 ms is unresolvable. Fit the same model on simulated Lat with known τ
  passed through the same `create_filter` and Hilbert to get the recoverable
  range and bias.
- *Figure.* Cross-validated R² against log τ (group median, subjects), with
  the simulated recovery band.

**C3. Misleading trials: wrong-hand lateralisation, then reversal.** Under
urgency gating early wrong evidence is forgotten within τ; an integrator
reverses late, later still the longer the misleading lead (Cisek et al.
2009; Thura & Cisek 2014). Transient wrong-hand activation is a classic LRP
signature (Eimer 1998, https://doi.org/10.3758/BF03209386). Accuracy on
misleading trials rises from 9 % to 60 % with waiting.
- *Test.* A new lock at the token where SP crosses 0.5 (from the token
  directions, cut from the go and decision locks). Lat referenced to the
  final correct side; cluster test for when it crosses 0; early-wrong vs
  late-correct commitments. Regress reversal latency on the size and duration
  of the misleading lead. ERF version: sensor double subtraction
  [(R−L | left) − (R−L | right)]/2 on MLC/MRC pairs in the go lock, expecting
  a wrong-hand LRF over tokens 1–4 that flips.
- *Expected.* Urgency gating: reversal about τ plus the filter delay after
  the crossing, independent of the lead's duration. Integration: later and
  lead-dependent.
- *Controls.* The crossing mixes elapsed times (include t); A5 (gaze to the
  leader); C9.
- *Figure.* Lat locked to the crossing, wrong vs correct; reversal latency
  against lead duration; the go-lock LRF by class.

**C4. Lateralised readiness field before commitment.** The LRF indexes
hand-specific preparation (Eimer 1998); central MEG sources follow
accumulated evidence (de Lange et al. 2010,
https://pmc.ncbi.nlm.nih.gov/articles/PMC6632985).
- *Test.* Decision lock. Sensor double subtraction on MLC/MRC and MLF/MRF;
  sources 4, 3a, 6d, 6a, 6v, 55b, FEF with S2. Cluster test against 0,
  jackknife onset, class × speed.
- *Controls.* Token direction correlates with the hand: check occipital
  pairs. A5.
- *Figure.* LRF by class; double-subtraction topography at −200 ms.

**C5. Frontoparietal evidence encoding with a latency gradient.** Lateralised
alpha and beta in IPS/PCeS, PMd/v and M1 encode each sample and the
accumulated belief, earliest in IPS (Murphy et al. 2021; Siegel et al. 2011,
https://pmc.ncbi.nlm.nih.gov/articles/PMC3047300/).
- *Test.* Token-locked single trials, L − R log power regressed on the
  token's direction and on SP before it. Groups: IPS (IP0, IP1, IP2, IPS1,
  LIP, VIP, MIP, AIP, 7PC), PMd (6a, 6d, FEF), PMv (6v, 6r), M1 (4). Cluster
  test 0–400 ms; half-maximum latencies bootstrapped over subjects. Then the
  sample-evidence β at its peak vertex-wise.
- *Controls.* Trust only differences between distant groups (C4); hand =
  hemifield (§2.5); token overlap.
- *Figure.* Encoding β per group with half-maximum markers; vertex map.

**C6. Neural kernel: primacy or recency.** A sample's neural weight predicts
its behavioural weight (Wyart et al. 2012,
https://pmc.ncbi.nlm.nih.gov/articles/PMC3975574/; Cheadle et al. 2014,
https://pmc.ncbi.nlm.nih.gov/articles/PMC4411568/). Behaviour shows primacy,
stronger in Fast for tokens 2–4. Two readings:
- **Neural primacy:** early evidence persists in the signal (Winkel et al.
  2014, https://doi.org/10.3758/s13423-013-0551-8).
- **Selection:** urgency gating with early commitments yields behavioural
  primacy from a recency-weighted neural kernel (Carland et al. 2015,
  https://doi.org/10.3758/s13423-015-0851-2).
- *Test.* Decision lock −200..0 ms, motor Lat regressed on the directions of
  tokens 1..K, separately for trials committing after token K (K = 5..8) and
  per speed. Include SP at commitment (the count is on screen).
- *Expected.* Selection: recency within each stratum, primacy only when
  pooled. Neural primacy: primacy within strata, larger in Fast for 2–4.
- *Figure.* Neural against behavioural kernel by stratum and speed.

**C7. A parietal build-up ramps to commitment.** The CPP builds at a rate set
by evidence and peaks at the response (O'Connell et al. 2012; Kelly &
O'Connell 2013); it is the P300 (Twomey et al. 2015,
https://doi.org/10.1111/ejn.12936). MEG places momentary evidence in PCC and
central regions (Bitzer et al. 2020,
https://www.frontiersin.org/journals/human-neuroscience/articles/10.3389/fnhum.2020.00009/full).
MEG is less sensitive to a radial CPP, so a null is weak.
- *Test.* Decision lock, S1, PCC (v23ab, d23ab, 31pd, 31pv, 7m, PCV) and
  parietal (7PC, 7AL, 7Am, IP1, IPS1, LIPv). Slope −500..−100 ms and level
  −100..0 ms by class × speed; vertex-wise slope map (S4).
- *Controls.* §2.2 (high-pass; confirm on D4); a response-locked ramp can be
  built from overlapping stimulus responses (Frömer et al. 2024,
  https://pmc.ncbi.nlm.nih.gov/articles/PMC11576515/; reply O'Connell et al.
  2025, https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12319789/), so report
  raw and deconvolved.
- *Figure.* PCC/parietal traces by class; slope and level bars.

### D. Token-level encoding

**D1. Token direction lateralises occipito-parietal responses.** Momentary
evidence appears in IP0, VIP, 7AL and PGp at 160–200 ms (Bitzer et al.
2020); visual gamma transiently encodes each sample (Murphy et al. 2021).
- *Test.* ERF: go lock, tokens 1–3; easy and misleading trials ending in the
  same choice have opposite early tokens. Lateralisation index of hemisphere
  RMS in V3A, V3B, V7, IPS1, IP0, VIP, 7AL, PGp, MT, MST; MLO/MRO, MLP/MRP
  sensors. Gamma: token-locked L − R regressed on direction, vertex-wise
  first, then V1, V2, V3, V3A, V3B, V6, IP0, IPS1, MT, MST.
- *Controls.* Gaze and microsaccades toward the moving token (A5); small
  misleading cells; C5 for gamma.
- *Figure.* Lateralisation index, misleading vs easy, with token ticks;
  fsaverage contralateral difference at 180 ms; gamma β map.

**D2. The late token response scales with decision relevance.** A
P3-like response at 250–500 ms scales with |ΔSP| or the posterior KL (Mars
et al. 2008, https://pmc.ncbi.nlm.nih.gov/articles/PMC6671727; Ruesseler et
al. 2023, https://elifesciences.org/articles/82823); sample gain depends on
relevance (Cheadle et al. 2014).
- *Test.* ERF (D1 derivative): token-locked overlap-corrected regression
  (`mne.decoding.ReceptiveField` or Unfold-style) on |ΔSP|, direction
  relative to the choice, token index and speed; S1 in IP0, IPS1, 7PC, 7AL,
  VIP, LIPv/d, PGp, v23ab, d23ab, 31pd, 7m; β maps (S4). Band power now:
  token-locked alpha desynchronisation and theta in V1–V4, MT, IPS1, 7PC on
  |ΔSP|, token index and pre/post commitment.
- *Expected.* Parietal β(|ΔSP|) > 0 at 300–500 ms; V1 does not scale.
- *Controls.* |ΔSP| correlates with token index and remaining tokens.

**D3. Reversal tokens.** A token that changes the leader distinguishes three
accounts: nothing beyond |ΔSP| (urgency gating); less response once evidence
has built up (de Lange et al. 2010; Cheadle et al. 2014); an extra response
scaled by change-point probability (Murphy et al. 2021). Midfrontal theta
signals conflict and the need for control (Cavanagh & Frank 2014,
https://pmc.ncbi.nlm.nih.gov/articles/PMC4112145/; Cohen & Donner 2013,
https://doi.org/10.1152/jn.00479.2013).
- *Test.* Add "reversal" and "consistent with leader" regressors to D2.
  Theta in a24pr, p24pr, a32pr, p32pr, 24dv, SCEF, 8BM on the same
  regressors with token-index dummies. Now: misleading vs ambiguous at
  tokens 4–5 (go lock).
- *Controls.* §2.3: every token evokes theta-band power, so compare reversal
  tokens with non-reversal tokens at the same index, and show V1 theta does
  not follow the regressor.

**D4. Token gain rises with time and in Fast.** Urgency as global gain
(Murphy et al. 2016); speed pressure boosts sensory encoding (Steinemann et
al. 2018); attention raises steady-state amplitude (Morgan et al. 1996,
https://pmc.ncbi.nlm.nih.gov/articles/PMC39354).
- *Test.* Per-token RMS, V1 vs parietal (IP0, IPS1, 7PC, LIPv): amplitude ~
  token × speed × level. Single-trial 5 Hz amplitude in sliding 1 s windows
  (D6): amplitude ~ time × speed + class; then decision time ~ early
  amplitude. Use the 5 Hz fundamental only.
- *Expected.* Parietal grows with token index and is larger in Fast; V1
  adapts (A2).

**D5. The 5 Hz tag spreads through the decision network.** Steady-state
responses tag processing of a periodic input (Norcia et al. 2015).
- *Test.* Go lock 0.2–1.0 s, SNR at 5/10/15 Hz (bin vs neighbours) per
  fsaverage vertex, sign-free; cluster test of SNR > 1. Latency from
  cross-correlation of rectified region responses; phase only after S2.
- *Controls.* Point spread from occipital cortex (C4).
- *Figure.* fsaverage 5 Hz SNR map; latency against hierarchy rank.

**D6. Commitment is phase-locked to token arrivals.** If a token triggers
each commitment, the 5 Hz token response survives decision locking; if
commitment phase is uniform it cancels (Wyart et al. 2012 on rhythmic
encoding).
- *Test.* Decision lock −1.5..−0.1 s, 5 Hz amplitude of V1–V4 averages (S3)
  and occipital sensors against the floor from go-lock amplitude/√nave or
  shuffled press times; a "last-token" response at fixed negative latency;
  behaviour: press time modulo 200 ms.
- *Why it matters.* Cheap, and it decides whether E2's bends can be
  last-token responses.

### E. Commitment and after

**E1. After commitment, motor regions stop encoding tokens.** PMd/M1 become
insensitive to evidence after commitment (Thura & Cisek 2014; Thura et al.
2022, https://pmc.ncbi.nlm.nih.gov/articles/PMC9754259/). The alternative,
post-decisional accumulation for confidence and error detection, keeps
encoding going (Murphy et al. 2015, https://elifesciences.org/articles/11946v1).
- *Test.* C5's encoding regression on tokens before vs after commitment, one
  speed at a time (§2.4), excluding tokens within ±200 ms of commitment
  (C8). Motor region and IPS (beta), V1–V3 and MT (gamma); token-locked
  ERFs with D1. Interaction region × epoch.
- *Expected.* Motor β → 0, V1 persists, IPS intermediate.
- *Figure.* Encoding β before vs after commitment, per region.

**E2. Neural commitment precedes the press; alpha bends before beta.**
Monkey commitment is about 280 ms before movement, and PMd bends about
200 ms before commitment (Thura et al. 2022). In the preprint, alpha source
trajectories bend at commitment and beta at the press (Thiery et al. 2022,
https://doi.org/10.1101/2022.06.14.494674).
- *Test.* Enter lock. Piecewise-linear fits per subject × region on quintile
  means (alpha in V1–V4, IPS1, 7PC and the motor region; beta in the motor
  region) and per trial on U and Lat; breakpoint latencies compared by
  paired bootstrap and against press − motor latency (about 450 ms by hand).
- *Controls.* Token-triggered decisions put the last token's response at a
  non-uniform time before the press (D6): include time since the last token
  or subtract a token template. C9.
- *Figure.* Trajectories with breakpoints and CIs; change-point histogram
  over the RT-run distribution.

**E3. Posterior alpha is suppressed while deliberating and rebounds after.**
Alpha falls with active sampling and rises with disengagement (Jensen &
Mazaheri 2010, https://doi.org/10.3389/fnhum.2010.00186).
- *Test.* Decision and enter locks, alpha in V1, V2, V3, V4, MT, MST, V3A,
  IPS1, 7PC, 7PL: −600..−200 ms against +300..+1000 ms per quintile.
  Expected: a larger rebound for later decisions.
- *Controls.* §2.3, §2.4: no Fast/Slow comparison after the press.

**E4. Post-decision tokens stay encoded; a Pe analogue locks to the
decisive token.** Accumulation continues after the choice and drives error
awareness (Murphy et al. 2015; Desender et al. 2021,
https://pmc.ncbi.nlm.nih.gov/articles/PMC8378845; Steinhauser & Yeung 2010,
https://pmc.ncbi.nlm.nih.gov/articles/PMC3073548). Here the error evidence
arrives with the tokens, so the Pe should lock to the token that makes the
chosen target lose.
- *Test.* With D0: post-press amplitude at the block's token rate in V1–V4
  and parietal regions, ambiguous/misleading vs easy within each speed.
  With D1/D2: error vs correct in the C7 regions, press-locked against
  decisive-token-locked. Theta in the midfrontal regions on tokens that
  contradict the choice.
- *Controls.* Early decisions leave more tokens: restrict to windows in which
  every cell still has tokens. Match class and decision time.

**E5. The post-movement beta rebound scales with success probability.** The
rebound scales with confidence in the motor plan and shrinks after errors
(Tan et al. 2016, https://pmc.ncbi.nlm.nih.gov/articles/PMC4737767/;
Torrecillos et al. 2015, https://pmc.ncbi.nlm.nih.gov/articles/PMC6795206/);
beta is bursty (Little et al. 2019,
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6795457/).
- *Test.* Enter lock +0.3..+1.5 s, contra motor beta; within each speed,
  rebound ~ SP at decision + correct + dt; dose–response over SP bins.
- *Controls.* §2.4 is decisive here (a 20 Hz post-press flicker would sit in
  this band): do A4 first, analyse Slow first. Remaining-token duration as a
  covariate.

### F. Fast vs Slow

Behaviour says Fast raises the urgency rate and the drift gain; the motor
ramp has the same rate in both. In monkeys, the fast block raises PMd/M1
baseline and leaves the slope (Thura & Cisek 2016), and appears as a block
shift present before the trial (Thura et al. 2022). F1 and F2 locate the
difference.

**F1. The block effect is an offset present before the trial.** Speed
emphasis raises pre-evidence motor preparation and pre-SMA/striatal baseline
(Steinemann et al. 2018; Forstmann et al. 2008,
https://doi.org/10.1073/pnas.0805903105; Boehm et al. 2014).
- *Test.* Absolute, subject-centred log power before the first token and the
  ramp's intercept and slope: U = b_cond + u·t, Δb against Δu, run-level
  permutation within subject. Regions first (motor, SCEF, 6ma, 6mp, 24dd,
  6a, FEF), then vertex-wise space-time clusters for any region with a rate
  difference. ERF: CNV before the first token (needs D4 with a trial-start
  lock; underpowered with the current 0.5 s window).
- *Expected.* Monkey pattern: lower beta in Fast before and during the
  trial, same slope.
- *Controls.* C10 (block order, drift, head position, trial excess); tonic
  muscle per run; the previous trial's post-press stream in the inter-trial
  interval (§2.4); faster deliberation saccades in Fast in monkeys (Thura et
  al. 2014, https://pmc.ncbi.nlm.nih.gov/articles/PMC6608489/), so A5.
- *Figure.* Fast − Slow pre-trial map (fsaverage, cluster test); intercept
  against slope per subject.

**F2. The stretch sits in evidence gain.** Hasty contexts amplify motor
activity for the chosen side (Derosiere et al. 2022,
https://pmc.ncbi.nlm.nih.gov/articles/PMC9017893/).
- *Test.* C1's slope of Lat on E by speed (speed × E × t); D4's token gain by
  speed; time scaling Slow(t) ≈ Fast(t/1.12) on Lat and on U, quantile
  matched.
- *Expected.* Fast E-slope 10–15 % larger with the same U slope: Lat
  stretches by 1.12, U does not.

### G. Monitoring, outcome and history

**G1. The medial-frontal response at the press tracks confidence.** Most
errors here are probabilistic: the leader was chosen and later lost. Error
markers scale with confidence (Boldt & Yeung 2015,
https://pmc.ncbi.nlm.nih.gov/articles/PMC4339357); the magnetic ERN source
is ACC (Keil et al. 2010, https://doi.org/10.1016/j.neuroimage.2009.12.055).
- *Test.* Enter lock 0–150 ms, S1 in a24pr, p24pr, 24dd, 32pr, a32pr, p32pr,
  SCEF, 8BM. Now: ambiguous vs easy. With D2: β(SP at commitment) vs
  β(correct). A classic ERN only for rare choices of the trailing target.
- *Controls.* Decision time; MEF leakage into SCEF/24dd (A1).

**G2. The outcome signal moves from feedback to the decisive token.** The
FRN encodes reward prediction error (Sambrook & Goslin 2015,
https://doi.org/10.1037/a0038195); the magnetic FRN has PCC then ACC sources
(Doñamayor et al. 2011, https://pmc.ncbi.nlm.nih.gov/articles/PMC6870214);
error theta and reward high beta (Cavanagh et al. 2010,
https://pmc.ncbi.nlm.nih.gov/articles/PMC2818688/; HajiHosseini & Holroyd
2015, https://pmc.ncbi.nlm.nih.gov/articles/PMC4538377/). The remaining
tokens reveal the outcome before feedback.
- *Test.* Feedback lock (D3): loss − win × speed in PCC (v23ab, d23ab, 31pd)
  and ACC (a24pr, p32pr), 200–500 ms; theta and high beta; decisive-token
  lock: loss − win.
- *Expected.* Small at feedback where the tokens were trackable, present at
  the decisive token.
- *Controls.* Errors over-represent misleading, ambiguous and long trials:
  match on both. Feedback delay after the last token, per block.

**G3. After an error: more caution.** PMd baseline changes after errors in
this task (Thura, Guberman & Cisek 2017,
https://pmc.ncbi.nlm.nih.gov/articles/PMC5288471/); post-error parietal
alpha suppression predicts slowing (Carp & Compton 2009,
https://doi.org/10.1111/j.1469-8986.2009.00777.x); error theta predicts
slowing (Cavanagh, Cohen & Allen 2009,
https://doi.org/10.1523/JNEUROSCI.4137-08.2009).
- *Test.* Pre-trial offset and ramp slope on trial n+1: beta in the motor
  region and SCEF/6ma, alpha in V1–V4, 7PC, IPS1, theta midfrontal. Mixed
  model power ~ previous correct + previous dt + previous class + previous
  condition + trial index; then whether the change predicts that trial's
  slowing.
- *Controls.* Errors cluster in early and misleading decisions; first trial
  of each run out.

**G4. Residual beta from the previous press drives alternation.** Motor
activity left by the previous response predicts alternation (Pape & Siegel
2016, https://pmc.ncbi.nlm.nih.gov/articles/PMC5059771/); pre-stimulus motor
lateralisation follows the previous choice (de Lange et al. 2013); history
biases accumulation (Urai et al. 2019, https://doi.org/10.7554/eLife.46331).
- *Test.* Go lock −0.5..0, motor Lat relative to the previous hand; logistic
  mixed model choice ~ evidence + previous choice + pre-go Lat; strongest on
  ambiguous trials. Does the bias stay constant or grow with t?
- *Controls.* Inter-trial interval varies with remaining tokens; the
  previous trial's post-press stream; previous correctness.

**G5. Pre-trial posterior alpha predicts behaviour.** Pre-stimulus alpha
shifts criterion and excitability (Samaha et al. 2020,
https://doi.org/10.1016/j.tics.2020.05.004).
- *Test.* Go lock −0.5..0 posterior alpha; log dt and correctness on alpha
  with class, condition and run. Combine with G3 and G4 (same window).

### H. Exploratory and confounded

**H1. Posterior alpha lateralisation follows the leading target** (Thut et
al. 2006, https://pmc.ncbi.nlm.nih.gov/articles/PMC6674607/; Wilming et al.
2020, https://pmc.ncbi.nlm.nih.gov/articles/PMC7547662/). Alpha L − R in V1–V4,
V3A, V7, IPS1, 7PL on E and on the token count on each side; lead–lag
against motor Lat. **Likely confounded:** more tokens on one side is more
stimulation there, gaze follows the leader, and hand = hemifield. Read only
if it survives the counts and A5.

**H2. Choice-predictive motor gamma before commitment** (Donner et al.
2009). Decision lock, gamma and high gamma contra − ipsi in the motor
region, onset against beta. **Likely underpowered;** C5.

**H3. Parietal/frontal gamma tracks evidence** (Polanía et al. 2014,
https://doi.org/10.1016/j.neuron.2014.03.014; Siegel et al. 2011).
Vertex-wise regression of decision-locked gamma on signed SP and |SP − 0.5|.
**Underpowered and exposed** to saccades, muscle and leakage.

### I. Deep structures

The preprint's subcortical claims were not reliable, and these structures
are mostly cortex here. Each deep hypothesis is a test against leakage.

**I1. Pallidal/striatal urgency against a leakage null.** The pallidum
carries urgency, not choice, in the tokens task (Thura & Cisek 2017,
https://doi.org/10.1016/j.neuron.2017.07.039; Thura et al. 2022); time
pressure raises striatal baseline (Forstmann et al. 2008); MEG sees deep
sources weakly (Pizzo et al. 2019,
https://pmc.ncbi.nlm.nih.gov/articles/PMC6393515/; Krishnaswamy et al. 2017,
https://pmc.ncbi.nlm.nih.gov/articles/PMC5715738/).
- *Test.* Project the observed cortical beta ramp (all 360 regions) through
  each subject's forward model (`mne.apply_forward`) with empty-room noise,
  invert with the same operator and compare observed − simulated in pallidum,
  putamen, caudate and thalamus; replicate with LCMV; check laterality. ERF:
  vector norm, regressed on neighbouring cortex, residual tested.
- *Honest prior.* No surplus over leakage; aseg cannot separate GPi, GPe or
  STN.

**I2. The preprint's subcortical speed–accuracy effect is neck muscle.**
High-frequency subcortical components tracked speed–accuracy in the preprint;
muscle is broadband at high frequency and the cerebellum and brainstem sit
by the neck (Andersen et al. 2020,
https://pmc.ncbi.nlm.nih.gov/articles/PMC7306153/).
- *Test.* Deep gamma/high gamma Fast − Slow in the nine tonic-muscle subjects
  against the 23, per stream, with per-run 110–140 Hz power as a covariate,
  plus the I1 null. If muscle: confined to the nine and scaling with muscle
  power. Cheap, and it answers the obvious reviewer question.

**I3. The cerebellar ramp is ipsilateral to the hand.** Ipsilateral
cerebellar desynchronisation is the opposite of leakage from contralateral
M1 (Andersen et al. 2020). Cerebellar Lat relative to the hand, observed
against the I1 simulation, with the I2 muscle split.

## 5. Where to look vertex-wise

Region-wise first everywhere; fsaverage for discovery where the region set
is not obvious or the effect is small and focal:
- **Maps for discovery:** F1 (Fast − Slow pre-trial and rate), C5 (evidence
  encoding β), D1 (token direction, gamma and ERF), D5 (5 Hz SNR), H3.
- **Maps for display** of region results: A1, A2, C7, G2.
- **Cost.** The fsaverage per-trial power is 24 TB. A vertex-wise test
  streams per-subject regression βs (or condition means) on Slurm, one task
  per subject × stream × band × lock, writing a vertex × time β map
  (about 8,196 × 2,100 floats) to `/project`; the group test runs locally on
  those maps. Per-condition and per-quintile fsaverage means are the same
  pattern and would be about 1 TB.

## 6. Derivatives to build

| | What | For | Cost |
|---|---|---|---|
| D0 | The post-press token schedule from the acquisition material or the stimulus channel | A4, E4, E5, F1, every post-press contrast | low; blocks the rest |
| D1 | Token-locked single-trial source ERFs (every token, metadata: index, direction relative to choice and correct target, SP, ΔSP, KL, reversal and decisive flags, time to commitment), overlap-corrected | D2, D3, E1, E4, G2 | high |
| D2 | Correct/error averages (trailing-target choices vs unlucky leaders), enter and decision | E4, G1 | low (`erp` with a new split) |
| D3 | Feedback-lock ERFs and band power (the epochs exist) | G2 | low |
| D4 | Re-epoching with a 0.05–0.1 Hz high-pass and a trial-start lock | B4, C7, F1 (ERF) | medium; preprocessing re-run of the filter step |
| D5 | ERFs by decision-time quintile within speed, matched nave | B2 (parietal), C7 | low |
| D6 | Single-trial 5 Hz amplitude and phase; token phase at commitment | D4, D6 | low |
| D7 | EOG channel averages per condition | A5 | low |
| D8 | ± noise floor per cell for every RMS measure | S3 | low |
| D9 | Induced band power (ERF subtracted) for the motor and visual regions | A3, every band result near 5/10/15/20 Hz | medium |

## 7. Order

1. **Verify and control:** D0, A1, A2, A3 (with D9), D6, I2.
2. **Region-wise, current data, strongest:** B1, B2, C3, C4, E3, E1 (Slow),
   F1 (band power), G3, G4.
3. **The models:** C1, C2 (with the filter simulation), C6, F2. This is the
   direct neural test of urgency gating against integration in humans; the
   only version so far is in monkeys.
4. **Vertex-wise discovery:** F1, C5, D1, D5.
5. **New derivatives:** D2/D3/D5 then G1, G2, C7; D1 then D2, D3, E4; D4 then
   B4 and the CNV.
6. **Last:** H1–H3, I1, I3.

**Most likely to give a strong, new, defensible result:** C2 + C3 (memory
length and the misleading reversal), C1 (multiplication with U measured
independently), F1 + F2 (where the Fast/Slow difference lives), B2 (which
account of the commitment level holds), E1 (evidence stops mattering after
commitment). I2 is the cheapest correction of the preprint.

## 8. Sources still to check

Cited from memory, through a secondary summary or by DOI only; confirm
before relying on them: Thura & Cisek 2014 and 2017; Cisek et al. 2009;
Thura et al. 2012; Jensen & Mazaheri 2010; Carland et al. 2015 and 2019;
Murphy et al. 2021 (bioRxiv; read in part); Yau et al. 2021 (cited through
bioRxiv); Unfold (bioRxiv). Several DOIs above were completed by hand from
the literature passes' PMC or repository links (Schurger 2012, Shibasaki &
Hallett 2006, Boehm 2014, Eimer 1998, Keil 2010, Sambrook & Goslin 2015,
Carp & Compton 2009, Cavanagh, Cohen & Allen 2009, Cohen & Donner 2013,
Donner 2009, de Lange 2013, Polanía 2014, Kelly 2021) and need the same check.

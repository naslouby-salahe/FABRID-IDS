# FABRID-IDS — complete, paragraph-anchored editorial revision

**Source:** November 6, 2026 iThenticate/Turnitin AI-writing report, manuscript pp. 1–12. Cross-check: October 11, 2026 report. This is a publication-quality language and scientific-clarity edit, **not** a predicted AI-detector score or a substitute for an author-led originality review.

## Apply these instructions exactly

1. Locate the paragraph by its section, opening words, and closing words. Replace the **entire indicated paragraph** with the replacement below, not individual PDF-wrapped lines. The last words in an anchor are descriptive locators; locate them in the PDF/source rather than trying to match PDF soft hyphens. Where a replacement spans two original paragraphs, it is labeled explicitly.
2. All replacements are complete and include any required citations. Retain scientific mathematical notation in the LaTeX source (`\alpha_k`, `w_k`, `B`, `\tau`, etc.). Keep all three equations, figures, tables, their numbering, data values, references [1]–[15], authors, affiliations and DOI. Do not substitute image-based text, truncate the chapter, obscure authorship, or replace prose with bullet lists to influence detection.
3. Maintain third-person scientific narration, do not add phrases such as “we,” “our,” or “this study.” Avoid duplicating the same limitations or findings in adjacent paragraphs. Do not invent research mechanisms, experiments, or results.
4. Sections that are intentionally **not** reworded include formula definitions, table rows and numerical values, reference metadata, headings, and figure data. Their inclusion within colored highlights does not alone warrant changing them.
5. Compile to the same Springer/IEREK template. Verify **at least ~4,000 content words excluding the first manuscript page and references**, a **maximum of 12 total manuscript pages**, minimum 6 eligible content pages, and **10–30 scholarly references**. Aim for a reasonable margin over 4,000 to account for counting rules; if any replacement shortens the paper too much, expand the technical reasoning with verified information from the existing implementation, never filler.
6. Generate a new similarity report and AI-content report on the final revised chapter through an authorized service. These reports cannot be inferred from prose changes, and a reduction to any particular percentage cannot be guaranteed.

## Exact, section-ordered replacements

### 01 — Abstract (manuscript p. 1; new report highlight group 1)

**Replace paragraph beginning** “High pooled attack-ranking performance does not ensure” **and ending** “pooled ranking metrics are close to one.”

**With:**

> The same anomaly scores can yield strong pooled ranking metrics but unreliable alerts after thresholds are selected for individual IoT devices. FABRID-IDS examines the allocation of a federation-wide nominal false-positive-rate (FPR) budget after training. Client-level targets are chosen jointly, then converted into thresholds using separate benign calibration records. The main evaluation covers nine N-BaIoT devices, ten detector seeds and five budget levels. A secondary experiment uses 19 CIC IoT-DIAD clients with a different feature representation and scorer. At B = 0.001, both unrestricted allocation objectives (Macro and Tail) produce a held-out federation FPR of 0.00170, approximately 1.696 times the budget. Conservative calibration lowers that FPR to 0.00036, but macro recall decreases from 0.3778 to 0.2445. Equal-FPR performs better in macro recall at several nominal budgets, although its measured FPR also exceeds the specified budget at several operating points. Pooled AUROC and AUPRC reach 0.9995 and 0.9999; nevertheless, the minimum client recall remains zero for every evaluated FABRID-family policy and budget on N-BaIoT. These results expose a practical limitation: allocating nominal FPR targets is not sufficient to control measured federation false alarms or maintain detection for every device.

**Reason:** Replace generic framing and duplicated ending with a concrete problem, exact protocol, quantitative findings, and bounded implication. The changes are editorial; no prediction of detector classification is made.

### 02 — Introduction, paragraph 1 (p. 2; highlight group 1)

**Replace from** “Federated IoT detection distributes model training” **through** “across clients.”

**With:**

> In a federated IoT intrusion-detection system, training produces an anomaly score, but an alert is issued only after that score crosses a threshold. The choice of threshold becomes difficult when participating devices have different benign-score distributions or when their attacks have different score profiles. Existing work examines federated detector training and both shared and device-specific thresholds [1–6]. It leaves a separate operational question: if the federation permits a limited average false-positive rate, which devices should receive larger or smaller nominal allowances? Equal allowances are straightforward to implement, but they do not account for how much attack recall each device gains when its threshold is relaxed.

### 03 — Introduction, paragraph 2 (p. 2; highlight group 1)

**Replace from** “FABRID-IDS treats this as alert-budget reallocation” **through** “realized-FPR guarantee.”

**With:**

> FABRID-IDS addresses that question after the scoring model is fixed. Using labeled validation attacks and benign frontier scores, the allocator chooses nominal client rates \(\alpha_k\) subject to \(\sum_k w_k\alpha_k\leq B\). Each device subsequently determines its alert threshold from a different benign calibration partition. The held-out evaluation therefore tests both the allocation decision and what happens when those rates are implemented using finite local calibration samples. The shared budget B restricts the nominal assignment; it is not a direct constraint on false alarms observed later.

### 04 — Introduction, paragraph 3 (p. 2; highlight group 1)

**Replace from** “We formulate client-specific FPR allocation” **through** “rather than relying only on federation averages.”

**With:**

> Three aspects are evaluated: the allocation of different nominal FPR allowances under one federation constraint; the gap between allocated rates and the thresholds produced by independent calibration; and the effect of each policy on average and worst-client attack recall. The N-BaIoT policies are compared on the same frozen scores and detector seeds. This paired design makes threshold selection the relevant policy difference without treating individual traffic rows as independent experimental repetitions.

### 05 — Related work, paragraph 1 (pp. 2–3; highlight groups 2–3)

**Replace from** “Federated IoT IDS research has mainly focused” **through** “target local FPRs.”

**With:**

> The distinction between detector training and alert policy is important when positioning FABRID-IDS. Federated IoT IDS papers have primarily developed learning architectures and aggregation strategies [1–3, 7]. Other work considers anomaly thresholds and detection at distributed edges [8, 9]. Laridi et al. [4] combine client-side summary statistics to calculate a shared anomaly threshold. Asiri et al. [5] derive local benign reconstruction-error quantiles and average the resulting thresholds. Fed-DTCN [6] instead uses benign hold-out samples to determine client-specific thresholds at prescribed local FPRs. These methods provide relevant thresholding baselines or context, but do not make the federation-wide division of nominal FPR allowances their optimization variable.

### 06 — Related work, paragraph 2 (p. 2; highlight group 4)

**Replace from** “Neyman–Pearson classification addresses” **through** “chronology or exchangeability.”

**With:**

> Error-rate constraints also appear in Neyman–Pearson classification [10], while conformal methods use calibration ranks to support finite-sample statements under explicit sampling assumptions [11]. Federated conformal work studies a related but different problem involving predictor sharing and calibration-score aggregation [12]. FABRID-IDS takes the calibration constructions as existing tools and places the allocation decision before them: the optimizer chooses the nominal target for each client, and a separate local sample determines the final threshold. The contribution is this federation-constrained allocation protocol and its measured behavior, not a new order-statistic theorem or a guarantee of realized FPR. Such a guarantee would need sampling conditions that the source-ordered N-BaIoT files do not establish.

### 07 — Related work, paragraph 3 (p. 2; highlight group 4)

**Replace from** “Training-time allocation is related” **through** “a shared limitation.”

**With:**

> Training-time fairness methods and agnostic federated learning modify the learning objective or the influence of client data [7]. FABRID-IDS does neither. Once training is complete, adjusting \(\alpha_k\) moves the decision threshold for device k without changing its benign or attack score rankings. This matters experimentally: threshold policies can be evaluated using identical stored scores, while any weaknesses of the scorer remain common to them. An allocation scheme cannot recover an attack that the fixed scorer consistently ranks below the required threshold without allowing a different operating point.

### 08 — Related work, paragraph 4 (starts p. 2, finishes p. 3; highlight groups 4 and 1)

**Replace from** “A common absolute threshold applies one cutoff” **through** “examined directly.”

**With:**

> Three operating choices are therefore kept distinct. A global absolute threshold applies one score cutoff everywhere. Equal-FPR allows local thresholds but specifies the same nominal rate at every client. FABRID-IDS also calibrates locally, while allowing the assigned rates to differ under the weighted federation constraint. Reporting the nominal assignments alongside held-out FPR and recall permits a direct comparison of the actual operating points rather than assuming that equal nominal budgets imply equal measured false alarms.

### 09 — Protocol and utility, paragraph 1 (p. 3; highlight groups 2–3)

**Replace from** “For each client k, let” **through** “until their respective stages.”

**With:**

> Let \(s_k(\cdot)\) be the frozen anomaly scorer for client k. Benign records are divided according to their released file order: 50% for training, 20% for estimating the allocation frontier, 10% for final threshold calibration and 20% for testing. For every retained attack subtype, the initial 20% is reserved for validation and the rest for testing. Only the frontier benign records and attack-validation subset are available when the allocator chooses nominal rates. The final-calibration subset is used afterward to turn those rates into thresholds, and the test subset is left untouched until evaluation. These proportions describe the available benchmark protocol, not a confirmed chronological deployment split.

### 10 — Protocol and utility, paragraph 2 around Eq. (1) (p. 3; highlight groups 4–5)

**Retain Eq. (1) unchanged. Replace the sentence before Eq. (1) and the paragraph after Eq. (1), up to** “across two subtypes.”

**With preceding text:**

> For a benign calibration sample of size n, sort its scores as \(x_{(1)}\leq\cdots\leq x_{(n)}\). The target rate \(\alpha\) is mapped to a threshold using the order-statistic rule in Eq. (1) [11].

**With following text:**

> Set \(\tau=+\infty\) for \(\alpha\leq0\), n = 0, or r > n; otherwise an observation is flagged only when its score is strictly greater than \(\tau\). The rule is applied first to the frontier benign sample at each candidate rate. The provisional threshold is then used to calculate \(u_k(\alpha)\), the mean validation recall over the eligible attack subtypes of client k. Attack labels are therefore required to construct the utility curve. The search grid contains 207 rates between 0 and 0.05 and includes all five evaluation budgets. Subtypes need at least 50 validation records, and a client is eligible only with at least 200 such records across two subtypes.

### 11 — Protocol and utility, paragraph 3 (p. 3; highlight group 6)

**Replace from** “Training records are used to fit” **through** “different score distributions.”

**With:**

> Model fitting and feature normalization use the benign training split of each client. Once an allocation has been selected, its provisional frontier thresholds are discarded and the order-statistic procedure is repeated on final-calibration benign scores. The test scores are not used at either selection stage. This separation prevents direct reuse of frontier observations for final threshold setting, but it does not force the frontier, calibration and test partitions to have identical score distributions. The released record order is particularly relevant because the underlying IoT features were computed from overlapping windows.

### 12 — Protocol and utility, paragraph 4 (p. 3; highlight group 6)

**Replace from** “Calibration uses discrete order statistics” **through** “different operating points.”

**With:**

> Nominal target rates are more finely spaced than the thresholds obtainable from a finite benign sample. Several adjacent values of \(\alpha_k\) may select the same order statistic, yielding the same threshold even though the optimizer records different rates. In that case, held-out decisions, recall and FPR also coincide because the scorer and test data are fixed. An allocated-rate change should consequently be interpreted as a distinct operating choice only if it produces a distinct calibrated cutoff. This finite-resolution effect becomes especially important for devices with small calibration partitions.

### 13 — Protocol and utility, paragraph 5 (p. 3; highlight group 6)

**Replace from** “For each client, attack recall is averaged” **through** “different measured false-alert rates.”

**With:**

> Attack recall is first computed separately for eligible subtypes and averaged within each device. The federation macro-recall endpoint gives each eligible device equal weight, regardless of the number of retained attack rows. The worst-client endpoint is the minimum of these device-level macro recalls. Both are reported with the held-out benign FPR. This is necessary because two policies can meet the same nominal federation constraint yet produce different false-alert rates after calibration, making an unqualified comparison of their recalls misleading.

### 14 — Allocation and calibration, opening and Eq. (2) (p. 4; highlight groups 1–2)

**Replace only the prose sentence** “FABRID-IDS reallocates the nominal alert budget by selecting client-specific grid rates subject to” **with:**

> The allocator selects one nominal rate from the candidate grid for each eligible client. Those choices are feasible only when the weighted sum is at most the specified federation budget:

**Keep Eq. (2) unchanged, including its upper-bound inequality.**

### 15 — Allocation and calibration, paragraph describing Macro, Tail, CVaR and rate resolution (p. 4; highlight group 3)

**Replace from** “Primary N-BaIoT results use” **through** “strict thresholding.”

**With:**

> The N-BaIoT implementation sets \(w_k=1/K\), so the budget represents an equal-device average rather than a row-weighted pooled rate. Allocation is solved as a one-hot mixed-integer program with HiGHS. The Macro objective maximizes the mean estimated client recall \(u_k\); Tail instead maximizes the mean of the lowest \(q=\lceil0.25K\rceil\) client utilities [13]. With nine devices, q = 3. The Tail objective thus averages the three weakest utilities rather than taking only the single minimum, and it is not numerically identical to empirical CVaR with a 0.25 tail fraction, which places a fractional weight on the third observation [13]. Each frontier sample has finite FPR resolution. For \(n_f\) benign frontier records, positive candidate rates below \(2/(n_f+1)\) are removed because they cannot create a positive empirical calibration-alert rate with the strict threshold rule.

### 16 — Allocation and calibration, interpretation of equal weights (p. 4; highlight group 3)

**Replace from** “Equal-client weighting makes federation FPR” **through** “receive a zero target.”

**With:**

> The equal-device definition should not be confused with pooling all benign records and calculating one false-positive fraction. The latter would assign more influence to the devices with larger files, even though retained row counts do not measure comparable traffic durations. Moreover, Eq. (2) defines a maximum expenditure, not a requirement to spend the full budget. The optimizer may leave part of B unused or set a client target to zero when the estimated utility does not justify spending additional allowance on that device.

### 17 — Allocation and calibration, solver priority and baseline (p. 4; highlight group 3)

**Replace from** “Macro maximizes estimated mean recall” **through** “through Eq. (3).”

**With:**

> Each optimization follows a specified order of priorities. Macro first maximizes estimated mean recall, then selects the smallest nominal expenditure among solutions accepted as equivalent under the solver tolerances. Tail places the lowest-three-client mean first, the overall estimated mean second and nominal expenditure third. Estimated utility variance and deterministic tie-breaking resolve the remaining choices. These calculations use frontier scores and labeled validation attacks only. Equal-FPR assigns B to every client, and all three unrestricted policies use the same final-calibration benign samples and Eq. (1). For the conservative Macro and Tail variants, the client allocations are retained but the final threshold ranks are determined by Eq. (3).

### 18 — Allocation and calibration, Tail's limitation (p. 4; adjacent to highlighted group 3)

**Replace from** “Tail optimizes the average utility” **through** “detect no attacks.”

**With:**

> A higher Tail objective does not necessarily imply better protection for the single weakest client. For example, an increase in the estimated recall of two members of the three-device lower group can improve its mean even if the remaining member continues to miss every evaluated attack. This is why the experiment reports minimum-client recall independently of the Tail optimization objective.

### 19 — Conservative calibration, text around Eq. (3) (pp. 4–5; highlight groups 4–5 on p. 4)

**Keep Eq. (3), its binomial quantile, and 0.95 unchanged. Replace the paragraph from** “Equation (3) searches the ranks” **through** “released source-order split.”

**With:**

> For each client, the conservative procedure searches integer ranks from 1 to n and uses the first rank meeting the binomial condition in Eq. (3). If no rank qualifies, or if n = 0 or \(\alpha=0\), the threshold is infinite and no alerts are issued. Alerts otherwise require \(s>\tau\), as in ordinary calibration. Under independently and identically distributed (i.i.d.) benign calibration and future scores, the rank construction provides a one-sided 95% population-FPR confidence statement for an individual client [10]. That statement concerns repeated calibration sampling; it is not a guarantee about the empirical FPR of the fixed, source-ordered test partition used here. The conservative variant is therefore evaluated as a sensitivity check rather than presented as validated federation-level control.

### 20 — Conservative calibration, ordinary versus conservative ranks (p. 5; highlight groups 1–2)

**Replace from** “The conservative rule usually selects” **through** “score distributions.”

**With:**

> Equation (3) generally requires a larger calibration rank than Eq. (1). A larger rank moves the threshold upward, making benign alerts less frequent but also allowing more attacks to fall below the cutoff. The distinction between the rules is how they handle finite-sample uncertainty: the ordinary construction uses the target quantile directly, whereas the conservative construction requires the additional binomial criterion. Neither method looks at test labels before choosing the threshold, and neither can compensate automatically if benign-score behavior differs between calibration and testing.

### 21 — Calibration interpretation and reporting (p. 5; highlight groups 2–3)

**Replace from** “An observed FPR above the assigned target” **through** “examined in the experiment.”

**With:**

> The allocation constraint and the reported test outcome answer different questions. A client can exceed its nominal rate on held-out data even when Eq. (2) was respected at allocation time. Equally, a low held-out FPR in one partition does not establish that the same threshold will remain safe on later traffic. For this reason, four quantities are kept separate: selected nominal client rate, final calibration rank, FPR within the calibration sample, and FPR on the held-out benign sample. Ordinary Macro, Tail and Equal-FPR differ in the assigned rates, not in the records or calibration formula used to turn those rates into operating thresholds.

### 22 — Zero-alert transition / Ecobee threshold (p. 5; highlight group 3)

**Replace from** “With strict alerts, α < 2/(n + 1)” **through** “held-out FPR shift.”

**With:**

> The smallest nonzero calibration-sample FPR is \(1/n\) when the n benign scores have no relevant ties. Because alerts use a strict greater-than comparison, targets below \(2/(n+1)\) either produce an out-of-range rank or select the maximum observed benign score; both cases yield zero calibration alerts. Consequently, a grid of finely spaced target rates contains choices with identical zero-alert behavior. The final-calibration subset for Ecobee has 1,311 benign records, which places this transition at approximately 0.001524. A nominal rate below that value cannot yield a positive empirical FPR on its calibration subset, although the same threshold may still flag benign observations in the separate test subset. The transition describes calibration resolution; the experiment does not isolate why Ecobee's held-out FPR differs.

### 23 — Frozen scorer (p. 5; highlight group 3, continuing group 4)

**Replace the whole paragraph from** “The primary scorer is the maximum” **through** “selected threshold differs.”

**With:**

> Each N-BaIoT device uses a frozen hybrid anomaly score constructed from two components: autoencoder reconstruction error and the temporal jitter feature HH jit L0.1 mean (f75). Before combining them, both components are expressed as percentile ranks against fixed frontier references; the final score is their maximum. The f75 feature and this scorer configuration were selected using the predeclared development-only ablations, not retuned for each threshold policy. The autoencoder has hidden layers [64, 16] and is trained by federated averaging for ten rounds, with three local epochs per round. Optimization uses Adam at learning rate 0.001, batch size 64 and mean-squared-error loss. Aggregation is weighted by the available training-row counts, while each client's normalization parameters come from its own training partition. Ten detector seeds provide the paired repetitions. Every evaluated threshold policy receives exactly the same scores within a seed; hence changes in threshold can change recall and FPR but not score-ranking measures such as AUROC or AUPRC.

### 24 — Experimental methodology, first two paragraphs (pp. 5–6; highlighted across page boundary)

**Replace from** “The primary experiment uses nine N-BaIoT devices” **through** “before the attack-score distributions are considered.”

**With:**

> N-BaIoT is the main evaluation dataset. Its nine devices serve as the natural client units, and each is tested at five nominal federation budgets: B = 0.001, 0.0025, 0.005, 0.010 and 0.020 (Table 1). The retained traffic includes benign observations and attacks from the BASHLITE and Mirai families [14]. Device-level sample sizes differ substantially: the benign count is 13,113 for Ecobee and 175,240 for the Philips baby monitor. Since final calibration receives 10% of each device's benign rows, those unequal file sizes produce unequal calibration precision. Macro recall is averaged over eligible attack subtypes within each device and then across devices; the minimum-client endpoint records the lowest device-level value. These definitions make the unit of comparison the device, not the number of packets or windows retained for that device.

### 25 — Experimental methodology, paired detector seeds (p. 6; highlight group 2)

**Replace from** “The five nominal budgets are” **through** “devices or collection periods.”

**With:**

> For any one detector seed, all policies are applied to identical trained models and score partitions. Comparing Macro against Equal-FPR, for example, therefore produces a within-seed difference rather than a comparison between unrelated detector runs. Ten seeds are available, and each contributes one observation for an endpoint at a given budget. The mean across seeds characterizes detector initialization and training variation on the fixed benchmark; it cannot be interpreted as a mean across newly collected clients, different collection periods or independently sampled traffic.

### 26 — Experimental methodology, FPR definition and split limitation (p. 6; highlight group 2)

**Replace from** “Each reported federation FPR is computed” **through** “devices.”

**With:**

> Held-out federation FPR uses the same device weights \(w_k=1/K\) as Eq. (2), applied to the benign test FPR measured separately for every device. This makes the nominal constraint and reported federation endpoint comparable in their weighting, although they are not guaranteed to have equal values. The test partition is derived from source-ordered files whose features were constructed from overlapping windows. It is not known to be an independent sample from the same distribution as calibration. Repeating detector training with different random seeds does not remove this dataset limitation.

### 27 — Experimental methodology, comparisons and paired differences (p. 6; highlight group 3)

**Replace from** “The prespecified comparisons are Macro” **through** “independent replicates.”

**With:**

> Two comparisons were specified for N-BaIoT: Macro minus Equal-FPR on federation macro recall, and Tail minus Equal-FPR on minimum-client recall. At each budget, the endpoint is evaluated separately for every trained seed before the baseline value is subtracted. The resulting ten differences preserve the pairing between policies. Traffic rows and attack subtypes are inputs to the client-level endpoint; they do not become additional statistical replicates for the tests.

### 28 — Experimental methodology, significance and intervals (p. 6; highlight group 3)

**Replace from** “We apply a two-sided exact sign-flip test” **through** “unseen devices.”

**With:**

> Inference uses a two-sided exact paired sign-flip test, enumerating all \(2^{10}=1{,}024\) sign assignments for the ten seed-level differences. A 95% pointwise confidence interval is also calculated using 50,000 bootstrap draws of those paired differences with replacement. The five budget-specific p-values within each endpoint family undergo Holm correction; the bootstrap intervals are not adjusted for simultaneous coverage. These quantities describe variation across the ten trained scorers on the same retained dataset. They do not estimate uncertainty for new device populations, newly collected benign traffic or repeated calibration periods.

### 29 — Experimental methodology, CIC IoT-DIAD secondary protocol (p. 6; highlight group 3)

**Replace from** “CIC IoT-DIAD is drawn” **through** “between the datasets.”

**With:**

> CIC IoT-DIAD, originating from the device-identification and anomaly-detection work of Rabbani et al. [15], is used as a secondary protocol check. Its 19 physical devices contribute 57 features, and a separate detector is trained for this dataset using reconstruction error rather than the N-BaIoT hybrid scorer. Benign partitions and the fraction of attacks reserved for validation follow the primary experiment. No N-BaIoT checkpoint or score ranking is transferred. Because the dataset, set of clients, representation and scorer all change together, this branch can show whether an FPR mismatch also occurs elsewhere, but it cannot determine which changed factor caused the difference.

### 30 — Results 5.1, first operating-point comparison (p. 6; not highlighted in November; keep unless necessary)

**Retain the original paragraph beginning** “At B = 0.001, unrestricted Macro and Tail” **and ending** “after local calibration.” **Do not rewrite it merely to match the surrounding sections.** It states the principal result with the requisite numerical values and experimental qualifier.

### 31 — Results 5.1, budget mismatch (p. 7; highlight group 1)

**Replace from** “The size of the mismatch varies” **through** “realized federation FPR.”

**With:**

> The held-out FPR does not track the nominal budget in the same way for every policy. At B = 0.0025, Macro and Tail operate at approximately 1.03B, only slightly above the allocated ceiling. Equal-FPR also overshoots at several budgets, showing that the mismatch is not unique to optimized rate allocation. Figure 1(a) shows the deviation from the nominal diagonal, while Fig. 1(b) places each measured macro-recall value against its corresponding realized FPR. The second view matters because equal values of B can produce different actual alert rates.

### 32 — Results 5.1, recall comparison at two budgets (p. 7; highlight group 1)

**Replace from** “At B = 0.005, Equal-FPR reaches” **through** “different measured FPRs.”

**With:**

> At nominal budget B = 0.005, the measured macro recalls are 0.8444 for Equal-FPR, 0.4644 for Macro and 0.4977 for Tail. Their realized federation FPRs are 1.45B, 0.71B and 1.29B, respectively. The gap is also visible at B = 0.020: Equal-FPR reaches 1.0000 macro recall, Macro reaches 0.7444 and Tail reaches 0.8444. The accompanying FPR ratios are 1.88B, 1.40B and 1.75B. Although Equal-FPR gives the highest recall at these reported operating points, it also incurs a higher measured FPR. These values do not provide a matched-FPR comparison or prove an advantage at a common false-alarm level.

### 33 — Results 5.1, frontier bootstrap sensitivity (p. 7; highlight group 1)

**Replace from** “We examined the stability of Macro allocations” **through** “repeated deployments.”

**With:**

> Allocation stability was checked by rerunning unrestricted Macro after bootstrap resampling the frontier benign scores and validation attack scores within each subtype. There are 500 such resamples per seed and budget. The trained scorer, the separate final-calibration scores and the test scores remain unchanged. For each seed–client–budget combination, the reported change frequency is the proportion of resamples whose assigned rate differs from the modal assignment over the 500 resamples. Averaging these frequencies across the ten seeds and nine clients gives 77.8%, 79.8%, 83.9%, 83.6% and 84.9% for ascending nominal budgets. Each value averages 90 client–seed frequencies; it is not a percentage of seeds whose rate changed. The result indicates considerable sensitivity of the optimization decision to the available frontier evidence, without representing repeated operational deployments or changes relative to the single original allocation.

### 34 — Figure 2 caption and paragraph (p. 8; highlight group 1)

**Keep the figure itself unchanged. Replace the caption** “Fig. 2 Original unrestricted Macro rates...” **with:**

> Fig. 2 Mean nominal rates allocated by unrestricted Macro to each N-BaIoT device, averaged across ten original detector-seed runs. The heatmap does not include the 500 frontier-bootstrap resamples.

**Replace from** “Figure 2 shows how Macro distributes” **through** “reported above.”

**With:**

> The heatmap makes the unequal rate assignments visible across the five budgets. At B = 0.020, the Samsung webcam and SimpleHome XCS7 1002 security camera receive the largest mean allocations, whereas Danmini Doorbell receives a smaller allowance. The cell values aggregate the ten primary seed runs; bootstrap-perturbed allocations belong only to the sensitivity experiment in Section 5.1. Thus the figure depicts the actual policy decisions evaluated on the held-out records, not an average over resampled hypothetical frontiers.

### 35 — Results 5.2, lower tail and Danmini (p. 8; highlight group 1)

**Replace from** “Tail optimizes the rounded three-client lower-tail mean” **through** “finite-sample transition.”

**With:**

> Tail is optimized for the mean of the three lowest client utilities, yet the observed minimum-client recall is zero for every FABRID-family policy and budget in Table 1. Danmini Doorbell explains this failure of the minimum endpoint: no attacks are detected under any evaluated unrestricted-policy operating point. Its final-calibration and held-out benign FPRs are also zero throughout the 100 combinations of unrestricted policy, seed and budget. Danmini's benign final-calibration sample contains 4,955 records, corresponding to a strict-threshold zero-alert transition at \(2/(4{,}955+1)=0.000404\). The assigned rates span both sides of that transition, but no evaluated unrestricted threshold recovers positive attack recall for this device.

### 36 — Results 5.2, Ecobee comparison (p. 8; paragraph adjacent to highlighted group)

**Replace from** “Unlike Danmini, Ecobee produces held-out benign alerts” **through** “evaluated unrestricted policies.”

**With:**

> Ecobee behaves differently despite its similarly restricted local calibration resolution. At B = 0.020, Macro assigns a mean nominal rate of 0.001164 to Ecobee. Its held-out benign FPR is 0.006672 and its macro attack recall is 0.6000, even though its final-calibration benign FPR is zero. Danmini, in contrast, combines zero held-out false alarms with zero attack detections. These outcomes show why a device's calibration FPR cannot be treated as a direct measurement of its later alert rate.

### 37 — Results 5.2, interpretation of client differences (p. 8; paragraph adjacent to highlighted group)

**Replace from** “Ecobee has 1,311 benign final-calibration records” **through** “individually.”

**With:**

> Ecobee has 1,311 final-calibration benign scores, compared with 4,955 for Danmini. These counts determine how finely the order statistic can represent the target rate, but sample size alone does not explain the observed difference in test alerts. A threshold that gives no calibration alerts can still flag benign test records. Since the released partitions are source ordered rather than known independent draws, the experiment cannot quantify how much of Ecobee's discrepancy is due to finite calibration resolution and how much reflects differences between the sampled score distributions.

### 38 — Results 5.3, primary comparisons (p. 9; not newly highlighted; preserve the evidence)

**Keep the paragraph beginning** “Macro and Equal-FPR are identical at B = 0.001” **and ending** “at those budgets.” **Do not change the tested differences or the significance interpretation.** Its mention of Equal-FPR's higher realized FPR is essential to avoiding a misleading headline based on recall alone.

### 39 — Results 5.3, score-ranking metrics (p. 9; highlight group 1)

**Replace the whole paragraph beginning** “Because allocation changes thresholds rather than anomaly scores” **and ending** “all five N-BaIoT budgets.”

**With:**

> Threshold reallocation cannot change the ordering of the stored anomaly scores, so AUROC and AUPRC are identical across policies within a detector seed. These metrics are computed after pooling the nine devices' held-out benign scores with attack-test scores from their eligible subtypes. The implementation calculates AUROC from score ranks and AUPRC by trapezoidal integration of the precision–recall curve. Averaged over ten seeds, the pooled scores yield AUROC 0.9995 and AUPRC 0.9999. This pooled calculation weights devices and attack subtypes by the number of their retained test rows rather than equally. In particular, the evaluation pool contains 1,041,078 attack rows and 111,188 benign rows, so attacks account for 90.35% of the pool. That prevalence raises the baseline level of precision–recall performance. Excellent pooled scores therefore do not establish equally good score separation for every device or acceptable recall at the selected local thresholds. In the same evaluation, minimum-client recall remains zero for all FABRID-family policy–budget combinations.

### 40 — Results 5.3, what the paired statistics do and do not compare (p. 9; highlight group 1)

**Replace from** “Table 2 compares policies at fixed nominal budgets” **through** “matching procedure.”

**With:**

> Table 2 reports paired policy differences at each stated value of B. Its confidence intervals quantify between-seed variation on the fixed N-BaIoT split, and the Holm-adjusted p-values address the five budget-wise tests within each endpoint family. Neither calculation adjusts the recall comparison to a common measured FPR. The five plotted points connected in Fig. 1(b) correspond to the five evaluated budgets; the lines are not additional experiments at intermediate false-alarm rates. Claims about recall at matched held-out FPR would require an explicitly designed comparison at those matched operating points.

### 41 — Results 5.4, CIC IoT-DIAD experiment (pp. 9–10; highlight groups 2 and 1)

**Replace from** “CIC IoT-DIAD provides a secondary protocol stress test” **through** “minimum-client recall from 0.0000 to 0.0251.”

**With:**

> The secondary test evaluates unrestricted Macro with ordinary order-statistic calibration on CIC IoT-DIAD. Table 3 summarizes ten separately trained scorers applied to 19 equally weighted physical-device clients, using reconstruction error as the anomaly score. The benign split proportions and attack-validation allocation are the same as in N-BaIoT, but the model is trained anew and no checkpoint or score ranking is shared across datasets. Across the five budget settings, the measured macro recall increases from 0.3614 to 0.5016 and held-out federation FPR increases from 0.0098 to 0.0311. Minimum-client recall ranges from zero to 0.0251. These are protocol-specific results; the 19-client configuration is not a matched extension of the N-BaIoT detector.

### 42 — Results 5.4, secondary interpretation (p. 10; immediately after highlighted text)

**Replace from** “The realized federation FPR also exceeds the nominal budget” **through** “common nominal client rate.”

**With:**

> The realized federation FPR exceeds the nominal constraint throughout this secondary test. The overshoot ratios at ascending budgets are 9.80, 4.72, 3.20, 2.13 and 1.56, even as the absolute measured FPR rises from 0.0098 to 0.0311. A decreasing ratio is therefore not evidence that false alarms become less frequent. Worst-client recall remains limited, reaching 0.0251 at the largest budget. Unlike the primary experiment, this branch does not include Equal-FPR under the same scorer and clients. It confirms that a nominal-to-measured FPR gap also appears in the secondary configuration, but does not show whether Macro offers better recall at a comparable false-alarm rate. Because multiple dataset and scorer properties change together, a shared cause cannot be assigned to both experiments.

### 43 — Discussion, opening interpretation (p. 10; mostly not highlighted but needed for continuity)

**Replace from** “On N-BaIoT, meeting the nominal allocation constraint” **through** “observed operating points.”

**With:**

> The N-BaIoT results separate two outcomes that can otherwise be conflated: whether the assigned rates satisfy the optimization constraint, and whether the calibrated thresholds meet the budget on held-out benign records. They do not always agree. Equal-FPR obtains higher macro recall at several budgets, but it also permits more measured false alarms at those operating points. Consequently, the evaluated results do not demonstrate that optimized reallocation is consistently better than the equal-rate baseline at a given measured FPR.

### 44 — Discussion, conservative calibration and Danmini (p. 10; highlight group 2)

**Replace from** “Equation (2) limits the weighted sum” **through** “nominal budget constraint.”

**With:**

> Equation (2) is enforced before final thresholds are calibrated. It limits the sum of assigned nominal rates, not the number of held-out benign scores that later cross each threshold. This distinction is especially visible at B = 0.001. Conservative calibration reduces the measured federation FPR compared with the ordinary rule, but macro attack recall also decreases. Neither calibration choice recovers attack detections for Danmini Doorbell under the tested Macro and Tail variants. All these allocations remain feasible under the nominal constraint, which is why feasibility alone is insufficient as an operational performance claim.

### 45 — Discussion, allocation/threshold stages and Ecobee (p. 10; highlight group 2)

**Replace from** “Macro and Tail estimate client utility from labeled validation attacks” **through** “not a label-free adaptive detector.”

**With:**

> Two separate samples affect the final policy. Frontier benign scores and labeled validation attacks define the estimated utility curve used to choose each client target. Another set of benign scores determines the final cutoff. For Ecobee at B = 0.020, the ordinary Macro threshold yields zero false positives on its calibration data but a mean held-out benign FPR of 0.006672. The source-ordered partitioning does not allow this difference to be attributed conclusively to a distribution shift or to calibration resolution alone. The attack labels used for the utility estimates also impose a deployment requirement: FABRID-IDS is an attack-informed allocation procedure applied after training, not an unsupervised online mechanism that discovers suitable client rates without attack examples.

### 46 — Discussion, solver runtimes (p. 11; highlight group 1)

**Replace from** “Across the five budgets, Macro solver times” **through** “end-to-end deployment timings.”

**With:**

> The measured allocation-solver cost is comparatively small: across budgets, Macro runs from 95.7 to 358.8 ms and Tail from 310.3 to 3883.1 ms. None of the recorded optimizations required solver fallback. These figures time the mathematical allocation step only. They exclude the cost of generating anomaly scores, communication, transmitting calibration information and processing alerts. They should therefore be used to characterize solver execution rather than interpreted as latency measurements for a deployed federated IDS.

### 47 — Discussion, study scope and secondary dataset (p. 11; highlight group 1)

**Replace from** “The primary evaluation covers one hybrid scorer” **through** “recover missed attacks.”

**With:**

> The evaluation spans one frozen hybrid scorer on nine N-BaIoT devices, ten training seeds and five budgets. CIC IoT-DIAD adds a second set of 19 clients, but also changes both input features and scoring method. Its results cannot isolate a cause shared with N-BaIoT, since those components were not varied independently. Other unresolved deployment questions include behavior under adversarial clients, formal privacy guarantees, the total communication and alert-handling cost, and whether improvements to the frozen scorer could address the attacks missed by the weakest devices. None of those interventions was measured in the experiments reported here.

### 48 — Discussion, Equal-FPR example at B = 0.020 (p. 11; highlight group 1)

**Replace from** “At B = 0.020, Equal-FPR achieves macro recall” **through** “same measured false-positive rate.”

**With:**

> The B = 0.020 comparison illustrates why an unqualified recall ranking would be misleading. Equal-FPR reaches macro recall 1.0000 with a measured federation FPR of 1.88B; Macro reaches recall 0.7444 with an FPR of 1.40B (Table 1). The two policies spend different amounts of the actual false-alert allowance, although they were evaluated against the same nominal budget. The result describes those two calibrated operating points and does not determine which policy would be preferable if both were required to operate at an identical measured FPR.

### 49 — Discussion, exact sign-flip test and multiplicity (p. 11; highlight group 1)

**Replace from** “For ten paired seeds, a two-sided exact sign-flip test” **through** “both endpoint families together.”

**With:**

> Statistical resolution is limited by the ten paired seeds. With \(2^{10}=1{,}024\) sign assignments, the smallest two-sided exact sign-flip p-value is \(2/1{,}024=0.001953125\), attained when all nonzero paired differences have the same sign. With five budget-level tests in a family, the smallest Holm-adjusted value appearing in Table 2 is 0.0098. At B = 0.001, the paired policy outcomes are identical, giving p = 1 and a zero-width interval. The intervals for minimum-client recall at B = 0.010 and 0.020 are also almost degenerate because Tail has zero recall across seeds and Equal-FPR varies only after the fourth decimal place. Holm adjustment is performed within each endpoint family; it does not jointly correct the two families, and the 95% bootstrap intervals remain pointwise.

### 50 — Discussion, conservative-rule probability statement (p. 11; highlight group 2)

**Replace from** “Equation (3) provides a per-client population-FPR statement” **through** “on the released data.”

**With:**

> The 95% statement associated with Eq. (3) applies to one client's population FPR under the required i.i.d. benign sampling assumptions. It is not a simultaneous 95% guarantee for the nine-device federation. Even if the assumptions were valid for each device, applying a union bound to nine separate 5% failure probabilities yields only \(1-9(0.05)=0.55\) as a lower bound on the probability that all nine nominal client limits hold together. The experiments do not establish the required sampling assumptions or an alternative federation-wide confidence construction, so no joint FPR guarantee is claimed for the released N-BaIoT split.

### 51 — Discussion, deployment implication (p. 11; immediately after highlighted group)

**Replace from** “A site that must keep federation FPR below a fixed ceiling” **through** “deployment-level control.”

**With:**

> An operational false-alarm ceiling would require thresholds calibrated against benign traffic representative of the intended monitoring period, followed by checks of achieved FPR after deployment. Optimizing nominal allowances on one benchmark partition cannot provide that assurance by itself. The conservative N-BaIoT experiment demonstrates a trade-off rather than a complete solution: at the tightest budget it reduces measured false alarms, while lowering macro recall and still missing attacks on Danmini. Testing an adaptive monitoring and recalibration mechanism under a credible deployment sequence remains outside the scope of the present source-order evaluation.

### 52 — Discussion, final summary of trade-offs (p. 11; adjacent to highlighted group)

**Replace from** “Within a detector seed, the same frozen scores” **through** “recalibrated samples.”

**With:**

> Keeping the scorer fixed gives the paired comparisons a clear interpretation: the policies change only the assigned thresholds. Their different realized FPRs must nevertheless accompany any differences in recall, because the trade-off depends on the achieved operating point. Tail's lower-group mean adds an independent consideration; a policy can improve two weak clients while leaving the single worst client undetected. For those reasons, the evaluation retains the allocated rate, held-out federation FPR, federation macro recall and minimum-client recall as separate quantities. Pairing by detector seed limits one source of training variation, but does not establish how those outcomes would transfer to new traffic collections or future recalibration windows.

### 53 — Conclusion, complete section (p. 12; highlight group 1)

**Replace the entire paragraph beginning** “FABRID-IDS divides a nominal federation-level FPR budget” **and ending** “The authors declare no competing interests.”

**With:**

> FABRID-IDS evaluates how a federation-level nominal false-positive allowance can be distributed across heterogeneous IoT clients after detector training. Local calibration converts the selected allowances into thresholds, and the resulting measured rates do not always remain within the original federation budget. On N-BaIoT at B = 0.001, unrestricted Macro and Tail both reach a measured federation FPR of 0.00170. Conservative calibration reduces this value to 0.00036, but at a cost to attack recall. Equal-FPR records higher macro recall at several nominal budgets while also operating at higher measured FPRs. None of the tested FABRID-family policies restores positive minimum-client recall across the nine-device evaluation. CIC IoT-DIAD shows a similar nominal-to-measured FPR discrepancy under a different configuration, although its experiment lacks a matched Equal-FPR comparator. The results support treating nominal allocation, finite-sample calibration and measured operating performance as separate decisions. Testing whether this framework can maintain a false-alarm ceiling in deployment will require representative calibration streams, monitoring and appropriate distributional assumptions. Implementation and reproducibility artifacts are available at https://doi.org/10.5281/zenodo.22113172. The authors declare no competing interests.

---

## Coverage verification against the November report

The report contains **32 contiguous blue-highlight groups** distributed across 12 manuscript pages. Some groups are line fragments of the same paragraph and others cross pages. Coverage by manuscript page:

| Manuscript page | November blue highlight groups | Covered by edit numbers |
|---|---:|---|
| 1 — Abstract | 1 | 01 |
| 2 — Introduction and Related work | 4 | 02–08 |
| 3 — Related work and Protocol | 6 | 08–13 |
| 4 — Allocation and calibration | 5 | 14–19 |
| 5 — Calibration, scorer, methodology | 4 | 20–24 |
| 6 — Experimental methodology | 3 | 24–29 |
| 7 — Results, FPR and stability | 1 | 31–33 |
| 8 — Heatmap and worst-client behavior | 1 | 34–37 |
| 9 — Ranking and CIC IoT-DIAD | 2 | 39–41 |
| 10 — Secondary results and Discussion | 2 | 41–45 |
| 11 — Discussion | 2 | 46–52 |
| 12 — Conclusion | 1 | 53 |

Sections 30 and 38 are intentionally retained to preserve accurate numerical results. Headings, mathematical equations, figure contents, table entries, and reference metadata are retained. All prose passages intersecting the new report's highlights are included in the replacements above, with adjacent paragraphs adjusted only when needed for coherent scientific interpretation.

## Manuscript compliance and final checks

- Existing manuscript in the Turnitin export is 12 manuscript pages, excluding the two report cover/overview pages. It is at the IEREK **maximum**: do not let final typesetting add another page.
- Its reference list contains 15 entries, within the required 10–30 range. Retain the existing references and validate DOI metadata independently if modifying them.
- The PDF-to-text extraction gives approximately 4,826 whitespace-delimited tokens between the start of the Introduction and the References heading, **before** removing headers, equations or other noneligible items. This is a rough audit figure **not** the publisher's eligible-word count. The exact content word count must be measured from the finalized chapter according to the template/editor's rules, with >4,000 eligible words and preferably a margin.
- The user's IEREK email requires 6–12 pages by its stated exclusions, approximately 4,000+ content words, similarity <=10–15% overall and <=5% from any single source, a similarity report **and** an AI-content report. No numerical AI threshold is given by the email.
- Verify data and statistical interpretation against the implementation and source-data artifacts. Do not treat scientific paraphrases as a substitute for experimental validation.
- For each replacement, confirm that citations point to the correct paper, especially [4]–[6], [10]–[13]. Verify the reference titles/DOIs on publisher records if any metadata look inconsistent.
- Run a PDF preflight after recompilation: 12 pages maximum; no overfull text; legends and captions visible; no figures/tables splitting claims; no shifted citations; no lost Greek symbols; complete author and corresponding-author fields.
- Preserve version history and author review. Final reports must be produced on exactly the final PDF after all edits, with a documented similarity-source breakdown. A Turnitin AI score is an estimate and no surgical replacement can guarantee it falls below an unstated threshold.

## Independent-source check: why October and November scores differ

Turnitin's official October 14, 2026 release notes confirm an English-language AI-writing detection model update after the October 11 report and before the November 6 report. Turnitin states previous reports are not retroactively updated and must be resubmitted to obtain a new-model score. That update can cause score differences even when some prose remains unchanged. The document was also rewritten between reports, making the two effects inseparable without rescanning the exact October file in the updated model. The 32% and 68% values are therefore not a controlled measurement of how much new text was AI-generated.

Primary sources:
- https://guides.turnitin.com/hc/en-us/articles/28294949544717-AI-writing-detection-model
- https://guides.ithenticate.com/hc/en-us/articles/27835303398541-AI-writing-detection-in-the-Similarity-Report
- https://www.springernature.com/gp/authors/publish-a-book/manuscript-guidelines

## Exact optional length reductions if the revised chapter exceeds 12 pages

Apply only as many of these sentence removals as necessary **after** rebuilding the original template. They target secondary explanations already supplied elsewhere and do not authorize deletion of results or limitations. Recheck the ~4,000-word eligibility rule after each set.

1. Edit 02: remove “Equal allowances are straightforward to implement, but they do not account for how much attack recall each device gains when its threshold is relaxed.”
2. Edit 04: remove “This paired design makes threshold selection the relevant policy difference without treating individual traffic rows as independent experimental repetitions.”
3. Edit 06: remove “Such a guarantee would need sampling conditions that the source-ordered N-BaIoT files do not establish.”
4. Edit 08: remove “Reporting the nominal assignments alongside held-out FPR and recall permits a direct comparison of the actual operating points rather than assuming that equal nominal budgets imply equal measured false alarms.”
5. Edit 11: remove “The released record order is particularly relevant because the underlying IoT features were computed from overlapping windows.” (Retained elsewhere in edit 26.)
6. Edit 12: remove “This finite-resolution effect becomes especially important for devices with small calibration partitions.”
7. Edit 13: remove “This is necessary because two policies can meet the same nominal federation constraint yet produce different false-alert rates after calibration, making an unqualified comparison of their recalls misleading.” (Retained in Results and Discussion.)
8. Edit 16: remove “The latter would assign more influence to the devices with larger files, even though retained row counts do not measure comparable traffic durations.” (The pooled-weighting distinction already appears in the preceding sentence.)
9. Edit 18: remove “This is why the experiment reports minimum-client recall independently of the Tail optimization objective.”
10. Edit 20: remove “The distinction between the rules is how they handle finite-sample uncertainty: the ordinary construction uses the target quantile directly, whereas the conservative construction requires the additional binomial criterion.” (Both rules remain explicit near Eqs. 1 and 3.)
11. Edit 22: remove “The transition describes calibration resolution; the experiment does not isolate why Ecobee's held-out FPR differs.” (Interpretation retained in edit 37.)
12. Edit 24: remove “These definitions make the unit of comparison the device, not the number of packets or windows retained for that device.”
13. Edit 25: remove “The mean across seeds characterizes detector initialization and training variation on the fixed benchmark; it cannot be interpreted as a mean across newly collected clients, different collection periods or independently sampled traffic.” (Retained in edit 28.)
14. Edit 29: remove “Because the dataset, set of clients, representation and scorer all change together, this branch can show whether an FPR mismatch also occurs elsewhere, but it cannot determine which changed factor caused the difference.” (Retained in edit 42.)
15. Edit 31: remove “The second view matters because equal values of B can produce different actual alert rates.”
16. Edit 34: remove “Thus the figure depicts the actual policy decisions evaluated on the held-out records, not an average over resampled hypothetical frontiers.”
17. Edit 36: remove “These outcomes show why a device's calibration FPR cannot be treated as a direct measurement of its later alert rate.”
18. Edit 40: remove “The five plotted points connected in Fig. 1(b) correspond to the five evaluated budgets; the lines are not additional experiments at intermediate false-alarm rates.”
19. Edit 47: remove “None of those interventions was measured in the experiments reported here.”
20. Edit 52: remove “Pairing by detector seed limits one source of training variation, but does not establish how those outcomes would transfer to new traffic collections or future recalibration windows.” (Retained in Methods.)

The ordered removals save roughly 300–400 words if all are applied; they are **conditional**, not automatic. Do not narrow figure margins, shrink fonts below the mandated template, or exclude citations or technical content from the word count through formatting tricks.

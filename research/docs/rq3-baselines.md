# RQ3 baselines: availability check (2026-10-18)

Candidate search-based baselines evaluated for RQ3: CoCoMagic, Causal-Fuzzer, AV-Fuzzer. "Verified" below means I opened the source myself. "Unverified" means it comes from a web-search summary only.

| Baseline | Public code? | Status |
|---|---|---|
| **AV-Fuzzer** (Li et al., ISSRE 2020) | Yes. `github.com/cclinus/AV-Fuzzer` (verified to exist; last push 2026-09-08, licence `NOASSERTION`, so check before reuse). Targets Apollo + LGSVL. | The original is bound to LGSVL and Apollo. |
| **AVFuzzer in Drivora** (`MingfeiCheng/Drivora`) | Yes. MIT, active (last push 2026-06), CARLA >= 0.9.12 (verified from README). A *prototype* GA with crossover, mutation, restart and local iterative search. The README states it is "not guaranteed to be fully identical to the original". Also ships Random, BehAVExplor, SAMOTA, and agent wrappers for InterFuser, LAV, TransFuser, SimLingo (our four non-Autoware ADSs). | Best practical route for AV-Fuzzer on CARLA. Must state in the paper that it is a re-implementation. |
| **Causal-Fuzzer** (Tang, Cheng et al., TSE) | **Not found.** There is a project page `sites.google.com/view/causal-fuzzer` titled "Artifacts". It is JavaScript-rendered, and I could not confirm whether it hosts code. No GitHub repo found by name search. Not in Drivora. | Open the page in a browser. If no code, email the authors (Mingfei Cheng is also the Drivora author). Otherwise re-implement from the paper. |
| **CoCoMagic** (Yousefizadeh, Gu, Briand, Nasr; arXiv 2509.16478) | **Not found.** No repo link in the paper abstract or from GitHub search. Evaluated on InterFuser in CARLA. | Email the authors for a replication package, or re-implement. |

## Protocol design considerations
1. **CoCoMagic is differential testing across ADS *versions*.** It searches for scenarios where two versions of a system diverge on metamorphic relations. It does not natively find safety violations of a single ADS, so "violations found" is not what it optimises. Comparing it on method-independent outcomes (violations, time to first violation) needs an adaptation, for example using two checkpoints of the same ADS as "versions", or fixing the metamorphic relation to a safety property. This choice must be stated in the fairness protocol and the threats to validity.
2. **AVFuzzer in Drivora is not the original.** Report it as a re-implementation. The cclinus original is LGSVL/Apollo-specific.
3. **Autoware is not in Drivora's agent corpus.** Running Drivora's fuzzers against Autoware needs a new agent wrapper, or running them through our own scenario runner.
4. **The common CARLA version differs.** Drivora supports 0.9.12+ and our Autoware bridge needs 0.9.15, while the harness has run on 0.9.16. Pinning every ADS and baseline to one version (0.9.15) would simplify the protocol, but needs the other four ADSs checked on 0.9.15.

## Suggested next step
Email the Causal-Fuzzer and CoCoMagic authors now (replies take days) while integrating Drivora's AVFuzzer.

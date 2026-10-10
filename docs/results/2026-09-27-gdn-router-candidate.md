# GDN/router 262k candidate panel — measured, not promoted

The private selected GDN metadata + router-dedup stack completed its final client on four Sparks. This portable document records returned values; it does not state that this repository tree was deployed, that the candidate was promoted, or that an unmeasured 1M configuration works. The exact returned file hashes and full sparkDash cell summary are in [candidate data](2026-09-27-gdn-router-candidate.json). The private prompts and raw dashboard jobs remain in the operator's evidence workspace.

| Check | Candidate | Prior accepted L2 | Reading |
|---|---:|---:|---|
| qeval c1 | **72/75** | 75/75 | Floor 72 passed; one truncation and three answer losses (`code_two_sum`, `math_m9`, `reason_r4`). |
| qeval c4 | **75/75** | 75/75 | No score loss. |
| Mean teacher-forced KL | **0.029189162** | 0.028836616 | 17 items, 6,618 positions; +0.000352546, manual numerical review required. |
| p99 KL | **0.267795120** | 0.283624938 | Lower in this panel; does not negate c1 answer losses. |
| Top-1 agreement | **0.923995165** | 0.925203989 | −0.001208824 on the same panel. |
| Prose c1 decode | **71.64 tok/s** | 70.19 tok/s | Median of five 256-output-token sparkDash jobs; +1.45 tok/s across sessions. |
| Prose c4 aggregate decode | **149.44 tok/s** | 152.89 tok/s | Median of three jobs; −3.45 tok/s across sessions. |
| Prose c16 aggregate decode | **305.59 tok/s** | 312.69 tok/s | One job per stack; descriptive only. |

The final sparkDash collector completed 20 unique jobs and 100 streams under its original short-prompt matrix. All stream token/reasoning and source checks passed. The exact-container guard returned success with a 30-second healthy idle recovery and stopped no containers. Those operational checks do not resolve whether the answer loss is acceptable. The previous strict 192-round CUDA-event screen measured target-start periods; its millisecond saving is a different benchmark and cannot be substituted for the sparkDash throughput values above.

The cross-session tok/s contrasts have no new confidence interval or causal isolation. The KLD checker ensures panel structure and reports the numerical values; it has no automatic promotion threshold.

A separate fresh prefill client completed two warmups and nine scored requests (three per nominal size). It measured elapsed time to the **first observable generated delta, including reasoning**, and divided actual API prompt tokens by that time:

| Nominal prompt | Candidate median input tok/s | Prior accepted median | Candidate median TTFT |
|---|---:|---:|---:|
| 16k | **2,202.12** | 2,184.83 | 7.587 s |
| 32k | **2,209.32** | 2,188.94 | 15.198 s |
| 64k | **2,196.90** | 2,178.06 | 30.638 s |

The root's returned-data audit and bounded exact-container guard passed; independent result review is pending at this draft snapshot. The API did not report cached-token counts, so this is a client-observed derived rate, not an isolated GPU kernel rate or a cache-zero proof. The roughly 0.8–0.9% cross-session differences have no significance or causal claim. [Exact summary and raw hashes](2026-09-27-gdn-router-candidate.json).

Long-context retrieval and capacity at 1M remain unqualified. The optional gather/L2 scout arm did not receive a valid held-out promotion. Production and public release decisions remain pending.

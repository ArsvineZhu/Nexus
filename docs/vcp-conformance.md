# Numerical reference checks

The conformance suite uses fixed outputs from [VCPToolBox 5adb12b](https://github.com/lioensky/VCPToolBox/commit/5adb12bb55e616e22bd0e0dbfdc13032c069941a) and the DTSC field and curve corpus archived in [Nous-Wave 412b062](https://github.com/Heptalogos-Devs/Nous-Wave/tree/412b062082afb75ccfae4c3f69fc606d3e3cc862/crates/retrieval/src/reference/dtsc). The latter records capture commit `e03b891d42055cdf3cab5dc961cc71dd5facd65a`.

| Calculation | Compared quantities |
| --- | --- |
| NNLS | Original-space projection, residual and column coefficients, including zero targets and rank deficiency. The VCP coefficients were captured in unit-column coordinates and are converted for comparison. |
| Local Cone | Residual ratios for opposed rays, conditional gain, tied rays and zero positive fit. |
| Hypernym | Parent selection and differentia against fixed input, with business concept IDs. |
| Warped Chunk | Four pinned exact distances within \(10^{-12}\), and equality of bounded exact search to exhaustive output. |
| River and wormhole | Frozen Ω and shared-budget composition using captured numeric terms. Public scenarios build fresh observations to check complete paths, canonical source roots, third-concept eligibility and edge attenuation. |
| Semantic deduplication | Pinned aligned-order result, explicit input-order selection on the differing-order case, and rejection of a zero input vector. |
| DTSC | FP64 field samples and geometry against the Nous corpus. Separate cases verify the explicit baseline, auxiliary rewards and trust decision. |

The VCP source capture runs the reference functions in a disposable checkout of the pinned revision. Its fixtures are [VCP observations and readouts](../crates/nexus/tests/fixtures/vcp_5adb12b.json), [NNLS cases](../crates/nexus-math/tests/fixtures/vcp-nnls.json), and [DTSC field](../crates/nexus-math/tests/fixtures/vcp-dtsc-field.json) and [curve](../crates/nexus-math/tests/fixtures/vcp-dtsc.json) samples.

~~~sh
python scripts/capture-vcp-golden.py temp/reference/VCPToolBox temp/vcp-golden.json
cargo test --workspace --all-features --locked
~~~

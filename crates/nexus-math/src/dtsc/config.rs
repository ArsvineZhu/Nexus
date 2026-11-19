// SPDX-License-Identifier: Apache-2.0

use nexus_core::{VcpError, VcpResult};

#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct DtscFieldConfig {
    pub min_field_tags: usize,
    pub min_field_entropy: f64,
    pub field_energy_mass_ratio: f64,
    pub max_field_nodes: usize,
    pub field_similarity_threshold: f64,
    pub field_kernel_exponent: f64,
    pub max_field_neighbors: usize,
    pub public_hub_floor: f64,
}

impl Default for DtscFieldConfig {
    fn default() -> Self {
        Self {
            min_field_tags: 3,
            min_field_entropy: 0.12,
            field_energy_mass_ratio: 0.95,
            max_field_nodes: 48,
            field_similarity_threshold: 0.5,
            field_kernel_exponent: 2.0,
            max_field_neighbors: 4,
            public_hub_floor: 0.35,
        }
    }
}

#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct DtscCurveConfig {
    pub strong_contact_threshold: f64,
    pub weak_contact_threshold: f64,
    pub min_closure_similarity: f64,
    pub position_decay: f64,
    pub min_geo_samples: usize,
}

impl Default for DtscCurveConfig {
    fn default() -> Self {
        Self {
            strong_contact_threshold: 0.16,
            weak_contact_threshold: 0.06,
            min_closure_similarity: 0.2,
            position_decay: 0.035,
            min_geo_samples: 3,
        }
    }
}

#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct DtscSparseConfig {
    pub enabled: bool,
    pub min_contacts: usize,
    pub min_conductance: f64,
    pub min_similarity: f64,
    pub min_potential: f64,
    pub min_closure: f64,
    pub pair_saturation: usize,
    pub max_relief: f64,
}

impl Default for DtscSparseConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            min_contacts: 3,
            min_conductance: 0.015,
            min_similarity: 0.48,
            min_potential: 0.08,
            min_closure: 0.2,
            pair_saturation: 3,
            max_relief: 0.55,
        }
    }
}

#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct DtscRewardConfig {
    pub alpha: f64,
    pub geo_reward_floor: f64,
    pub geo_reward_saturation: f64,
    pub direct_bonus_cap: f64,
    pub structural_bonus_cap: f64,
    pub thematic_bonus_cap: f64,
    pub structural_continuity_min: f64,
    pub thematic_min_potential: f64,
    pub thematic_max_isolated_ratio: f64,
    pub direct_semantic_min_potential: f64,
    pub direct_semantic_saturation: f64,
    pub direct_semantic_min_contacts: usize,
    pub direct_confidence_floor: f64,
}

impl Default for DtscRewardConfig {
    fn default() -> Self {
        Self {
            alpha: 0.35,
            geo_reward_floor: 0.015,
            geo_reward_saturation: 0.25,
            direct_bonus_cap: 0.18,
            structural_bonus_cap: 0.10,
            thematic_bonus_cap: 0.035,
            structural_continuity_min: 0.08,
            thematic_min_potential: 0.08,
            thematic_max_isolated_ratio: 0.65,
            direct_semantic_min_potential: 0.16,
            direct_semantic_saturation: 0.35,
            direct_semantic_min_contacts: 2,
            direct_confidence_floor: 0.35,
        }
    }
}

#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct DtscTrustConfig {
    pub min_candidate_coverage: f64,
    pub min_geo_score: f64,
    pub min_geo_spread: f64,
    pub min_strong_evidence: f64,
}

impl Default for DtscTrustConfig {
    fn default() -> Self {
        Self {
            min_candidate_coverage: 0.2,
            min_geo_score: 0.01,
            min_geo_spread: 0.03,
            min_strong_evidence: 1.0,
        }
    }
}

#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct DtscIdentityConfig {
    pub min_potential: f64,
    pub min_specificity: f64,
    pub min_tag_chunk_closure: f64,
    pub min_strength: f64,
    pub floor_cap: f64,
    pub floor_exponent: f64,
}

impl Default for DtscIdentityConfig {
    fn default() -> Self {
        Self {
            min_potential: 0.8,
            min_specificity: 0.55,
            min_tag_chunk_closure: 0.35,
            min_strength: 0.55,
            floor_cap: 0.018,
            floor_exponent: 1.25,
        }
    }
}

#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct DtscAuxConfig {
    pub max_aux_bonus: f64,
    pub direct_floor_cap: f64,
    pub structural_floor_cap: f64,
    pub thematic_floor_cap: f64,
    pub min_fused_score: f64,
    pub min_closure_score: f64,
    pub min_class_evidence: f64,
    pub floor_exponent: f64,
    pub identity_anchor: Option<DtscIdentityConfig>,
}

impl Default for DtscAuxConfig {
    fn default() -> Self {
        Self {
            max_aux_bonus: 0.018,
            direct_floor_cap: 0.018,
            structural_floor_cap: 0.012,
            thematic_floor_cap: 0.006,
            min_fused_score: 0.12,
            min_closure_score: 0.55,
            min_class_evidence: 0.1,
            floor_exponent: 1.5,
            identity_anchor: None,
        }
    }
}

#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct DtscConfig {
    pub field: DtscFieldConfig,
    pub scoring: DtscScoringConfig,
}

#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct DtscScoringConfig {
    pub evidence: DtscEvidenceConfig,
    pub readout: DtscReadoutConfig,
}

#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct DtscEvidenceConfig {
    pub curve: DtscCurveConfig,
    pub sparse: DtscSparseConfig,
    pub reward: DtscRewardConfig,
    pub auxiliary: Option<DtscAuxConfig>,
}

#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct DtscReadoutConfig {
    pub trust: DtscTrustConfig,
    pub top_k: usize,
    pub fallback_on_low_trust: bool,
}

impl Default for DtscReadoutConfig {
    fn default() -> Self {
        Self {
            trust: DtscTrustConfig::default(),
            top_k: 0,
            fallback_on_low_trust: true,
        }
    }
}

fn require(condition: bool, name: &str, constraint: &str) -> VcpResult<()> {
    if condition {
        Ok(())
    } else {
        Err(VcpError::InvalidInput {
            stage: "dtsc.config",
            detail: format!("{name} must {constraint}"),
        })
    }
}

fn range(name: &str, value: f64, lower: f64, upper: f64) -> VcpResult<()> {
    require(
        (lower..=upper).contains(&value),
        name,
        &format!("be in [{lower}, {upper}]"),
    )
}

impl DtscFieldConfig {
    pub fn validate(&self) -> VcpResult<()> {
        require(
            self.min_field_tags > 0 && self.min_field_tags <= self.max_field_nodes,
            "min_field_tags",
            "be positive and at most max_field_nodes",
        )?;
        require(
            self.max_field_neighbors > 0,
            "max_field_neighbors",
            "be positive",
        )?;
        range("min_field_entropy", self.min_field_entropy, 0.0, 1.0)?;
        range(
            "field_energy_mass_ratio",
            self.field_energy_mass_ratio,
            0.5,
            1.0,
        )?;
        range(
            "field_similarity_threshold",
            self.field_similarity_threshold,
            -1.0,
            1.0,
        )?;
        range("public_hub_floor", self.public_hub_floor, 0.05, 1.0)?;
        require(
            self.field_kernel_exponent.is_finite() && self.field_kernel_exponent > 0.0,
            "field_kernel_exponent",
            "be finite and positive",
        )
    }
}

impl DtscConfig {
    pub fn validate(&self) -> VcpResult<()> {
        self.field.validate()?;
        self.scoring.evidence.validate()?;
        self.scoring.readout.validate()
    }
}

impl DtscEvidenceConfig {
    pub fn validate(&self) -> VcpResult<()> {
        let c = &self.curve;
        range(
            "curve.strong_contact_threshold",
            c.strong_contact_threshold,
            0.0,
            1.0,
        )?;
        range(
            "curve.weak_contact_threshold",
            c.weak_contact_threshold,
            0.0,
            c.strong_contact_threshold,
        )?;
        range(
            "curve.min_closure_similarity",
            c.min_closure_similarity,
            -1.0,
            1.0,
        )?;
        require(
            c.position_decay.is_finite() && c.position_decay >= 0.0,
            "curve.position_decay",
            "be finite and nonnegative",
        )?;
        require(
            c.min_geo_samples > 0,
            "curve.min_geo_samples",
            "be positive",
        )?;
        let s = &self.sparse;
        if s.enabled {
            require(
                s.min_contacts >= 2,
                "sparse.min_contacts",
                "be at least two",
            )?;
            require(
                s.pair_saturation > 0,
                "sparse.pair_saturation",
                "be positive",
            )?;
            range("sparse.min_conductance", s.min_conductance, 0.0, 1.0)?;
            range("sparse.min_similarity", s.min_similarity, -1.0, 1.0)?;
            range("sparse.min_potential", s.min_potential, 0.0, 1.0)?;
            range("sparse.min_closure", s.min_closure, 0.0, 1.0)?;
            range("sparse.max_relief", s.max_relief, 0.0, 0.8)?;
        }
        let r = &self.reward;
        for (name, value) in [
            ("reward.alpha", r.alpha),
            ("reward.direct_bonus_cap", r.direct_bonus_cap),
            ("reward.structural_bonus_cap", r.structural_bonus_cap),
            ("reward.thematic_bonus_cap", r.thematic_bonus_cap),
            ("reward.geo_reward_floor", r.geo_reward_floor),
            (
                "reward.structural_continuity_min",
                r.structural_continuity_min,
            ),
            ("reward.thematic_min_potential", r.thematic_min_potential),
            (
                "reward.thematic_max_isolated_ratio",
                r.thematic_max_isolated_ratio,
            ),
            (
                "reward.direct_semantic_min_potential",
                r.direct_semantic_min_potential,
            ),
            ("reward.direct_confidence_floor", r.direct_confidence_floor),
        ] {
            range(name, value, 0.0, 1.0)?;
        }
        range(
            "reward.geo_reward_saturation",
            r.geo_reward_saturation,
            r.geo_reward_floor + 1e-6,
            1.0,
        )?;
        range(
            "reward.direct_semantic_saturation",
            r.direct_semantic_saturation,
            r.direct_semantic_min_potential + 1e-6,
            1.0,
        )?;
        require(
            r.direct_semantic_min_contacts > 0,
            "reward.direct_semantic_min_contacts",
            "be positive",
        )?;
        if let Some(a) = &self.auxiliary {
            range("auxiliary.max_aux_bonus", a.max_aux_bonus, 0.0, 0.05)?;
            for (name, value) in [
                ("auxiliary.min_fused_score", a.min_fused_score),
                ("auxiliary.min_closure_score", a.min_closure_score),
                ("auxiliary.min_class_evidence", a.min_class_evidence),
                ("auxiliary.direct_floor_cap", a.direct_floor_cap),
                ("auxiliary.structural_floor_cap", a.structural_floor_cap),
                ("auxiliary.thematic_floor_cap", a.thematic_floor_cap),
            ] {
                range(name, value, 0.0, 1.0)?;
            }
            range("auxiliary.floor_exponent", a.floor_exponent, 0.5, 4.0)?;
            if let Some(id) = &a.identity_anchor {
                for (name, value) in [
                    ("identity.min_potential", id.min_potential),
                    ("identity.min_specificity", id.min_specificity),
                    ("identity.min_tag_chunk_closure", id.min_tag_chunk_closure),
                    ("identity.min_strength", id.min_strength),
                    ("identity.floor_cap", id.floor_cap),
                ] {
                    range(name, value, 0.0, 1.0)?;
                }
                range("identity.floor_exponent", id.floor_exponent, 0.5, 4.0)?;
            }
        }
        Ok(())
    }
}

impl DtscReadoutConfig {
    pub fn validate(&self) -> VcpResult<()> {
        let trust = &self.trust;
        range(
            "trust.min_candidate_coverage",
            trust.min_candidate_coverage,
            0.0,
            1.0,
        )?;
        range("trust.min_geo_score", trust.min_geo_score, 0.0, 1.0)?;
        range("trust.min_geo_spread", trust.min_geo_spread, 0.0, 1.0)?;
        require(
            trust.min_strong_evidence.is_finite() && trust.min_strong_evidence >= 0.0,
            "trust.min_strong_evidence",
            "be finite and nonnegative",
        )
    }
}

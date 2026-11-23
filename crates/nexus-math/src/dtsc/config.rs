// SPDX-License-Identifier: Apache-2.0

use crate::{Error, Result};

#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct DtscFieldConfig {
    pub min_field_tags: usize,
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
    pub min_field_entropy: f64,
    pub min_candidate_coverage: f64,
    pub min_geo_score: f64,
    pub min_geo_spread: f64,
    pub min_strong_evidence: f64,
}

impl Default for DtscTrustConfig {
    fn default() -> Self {
        Self {
            min_field_entropy: 0.12,
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
    pub geometry: Option<DtscGeometryAuxConfig>,
    pub identity: Option<DtscIdentityConfig>,
}

impl Default for DtscAuxConfig {
    fn default() -> Self {
        Self {
            max_aux_bonus: 0.018,
            geometry: Some(DtscGeometryAuxConfig::default()),
            identity: None,
        }
    }
}

#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct DtscGeometryAuxConfig {
    pub direct_floor_cap: f64,
    pub structural_floor_cap: f64,
    pub thematic_floor_cap: f64,
    pub min_fused_score: f64,
    pub min_closure_score: f64,
    pub min_class_evidence: f64,
    pub floor_exponent: f64,
}

impl Default for DtscGeometryAuxConfig {
    fn default() -> Self {
        Self {
            direct_floor_cap: 0.018,
            structural_floor_cap: 0.012,
            thematic_floor_cap: 0.006,
            min_fused_score: 0.12,
            min_closure_score: 0.55,
            min_class_evidence: 0.1,
            floor_exponent: 1.5,
        }
    }
}

#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct DtscConfig {
    pub field: DtscFieldConfig,
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
    pub fallback_on_low_trust: bool,
    pub correction_gain: f64,
}

impl Default for DtscReadoutConfig {
    fn default() -> Self {
        Self {
            trust: DtscTrustConfig::default(),
            fallback_on_low_trust: true,
            correction_gain: 1.0,
        }
    }
}

fn require(condition: bool, field: &'static str, constraint: &str) -> Result<()> {
    if condition {
        Ok(())
    } else {
        Err(Error::config(field, constraint))
    }
}

fn range(field: &'static str, value: f64, start: f64, end: f64) -> Result<()> {
    require(
        (start..=end).contains(&value),
        field,
        &format!("value must be in [{start}, {end}]"),
    )
}

fn nonnegative(field: &'static str, value: f64) -> Result<()> {
    require(
        value.is_finite() && value >= 0.0,
        field,
        "value must be finite and nonnegative",
    )
}

fn greater(field: &'static str, value: f64, min: f64) -> Result<()> {
    require(
        value.is_finite() && value > min,
        field,
        &format!("value must be finite and greater than {min}"),
    )
}

fn fraction(field: &'static str, value: f64) -> Result<()> {
    require(
        (0.0..1.0).contains(&value),
        field,
        "value must be in [0, 1)",
    )
}

impl DtscFieldConfig {
    pub(super) fn validate(&self) -> Result<()> {
        require(
            self.min_field_tags > 0 && self.min_field_tags <= self.max_field_nodes,
            "min_field_tags",
            "must be positive and at most max_field_nodes",
        )?;
        require(
            self.max_field_neighbors > 0,
            "max_field_neighbors",
            "must be positive",
        )?;
        require(
            self.field_energy_mass_ratio > 0.0 && self.field_energy_mass_ratio <= 1.0,
            "field_energy_mass_ratio",
            "must be in (0, 1]",
        )?;
        require(
            (-1.0..1.0).contains(&self.field_similarity_threshold),
            "field_similarity_threshold",
            "must be in [-1, 1)",
        )?;
        range("public_hub_floor", self.public_hub_floor, 0.0, 1.0)?;
        require(
            self.field_kernel_exponent.is_finite() && self.field_kernel_exponent >= 0.25,
            "field_kernel_exponent",
            "must be finite and at least 0.25",
        )
    }
}

impl DtscEvidenceConfig {
    pub(super) fn validate(&self) -> Result<()> {
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
        nonnegative("curve.position_decay", c.position_decay)?;
        require(
            c.min_geo_samples > 0,
            "curve.min_geo_samples",
            "must be positive",
        )?;

        let s = &self.sparse;
        require(
            s.min_contacts >= 2,
            "sparse.min_contacts",
            "must be at least two",
        )?;
        require(
            s.pair_saturation > 0,
            "sparse.pair_saturation",
            "must be positive",
        )?;
        nonnegative("sparse.min_conductance", s.min_conductance)?;
        range("sparse.min_similarity", s.min_similarity, -1.0, 1.0)?;
        range("sparse.min_potential", s.min_potential, 0.0, 1.0)?;
        range("sparse.min_closure", s.min_closure, -1.0, 1.0)?;
        range("sparse.max_relief", s.max_relief, 0.0, 1.0)?;

        let r = &self.reward;
        for (name, value) in [
            ("reward.alpha", r.alpha),
            ("reward.direct_bonus_cap", r.direct_bonus_cap),
            ("reward.structural_bonus_cap", r.structural_bonus_cap),
            ("reward.thematic_bonus_cap", r.thematic_bonus_cap),
            ("reward.geo_reward_floor", r.geo_reward_floor),
        ] {
            nonnegative(name, value)?;
        }
        for (name, value) in [
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
        greater(
            "reward.geo_reward_saturation",
            r.geo_reward_saturation,
            r.geo_reward_floor,
        )?;
        greater(
            "reward.direct_semantic_saturation",
            r.direct_semantic_saturation,
            r.direct_semantic_min_potential,
        )?;
        require(
            r.direct_semantic_min_contacts > 0,
            "reward.direct_semantic_min_contacts",
            "must be positive",
        )?;

        if let Some(aux) = &self.auxiliary {
            nonnegative("auxiliary.max_aux_bonus", aux.max_aux_bonus)?;
            if let Some(g) = &aux.geometry {
                nonnegative("auxiliary.geometry.direct_floor_cap", g.direct_floor_cap)?;
                nonnegative(
                    "auxiliary.geometry.structural_floor_cap",
                    g.structural_floor_cap,
                )?;
                nonnegative(
                    "auxiliary.geometry.thematic_floor_cap",
                    g.thematic_floor_cap,
                )?;
                fraction("auxiliary.geometry.min_fused_score", g.min_fused_score)?;
                fraction("auxiliary.geometry.min_closure_score", g.min_closure_score)?;
                fraction(
                    "auxiliary.geometry.min_class_evidence",
                    g.min_class_evidence,
                )?;
                greater("auxiliary.geometry.floor_exponent", g.floor_exponent, 0.0)?;
            }
            if let Some(i) = &aux.identity {
                fraction("auxiliary.identity.min_potential", i.min_potential)?;
                fraction("auxiliary.identity.min_specificity", i.min_specificity)?;
                fraction(
                    "auxiliary.identity.min_tag_chunk_closure",
                    i.min_tag_chunk_closure,
                )?;
                fraction("auxiliary.identity.min_strength", i.min_strength)?;
                nonnegative("auxiliary.identity.floor_cap", i.floor_cap)?;
                greater("auxiliary.identity.floor_exponent", i.floor_exponent, 0.0)?;
            }
        }
        Ok(())
    }
}

impl DtscReadoutConfig {
    pub(super) fn validate(&self) -> Result<()> {
        nonnegative("correction_gain", self.correction_gain)?;
        let t = &self.trust;
        range("trust.min_field_entropy", t.min_field_entropy, 0.0, 1.0)?;
        range(
            "trust.min_candidate_coverage",
            t.min_candidate_coverage,
            0.0,
            1.0,
        )?;
        nonnegative("trust.min_geo_score", t.min_geo_score)?;
        nonnegative("trust.min_geo_spread", t.min_geo_spread)?;
        nonnegative("trust.min_strong_evidence", t.min_strong_evidence)
    }
}

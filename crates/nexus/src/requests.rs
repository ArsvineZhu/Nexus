// SPDX-License-Identifier: CC-BY-NC-SA-4.0
use crate::{
    Candidates, ConceptSearch, DtscConfig, DtscOptions, EvaluateOptions, Model, ObserveConfig,
    Query, Ranking, Result, RiverConfig, RiverOptions, RunOptions,
};

pub struct RiverRequestOptions<'a> {
    pub observe: ObserveConfig,
    pub river: RiverConfig,
    pub limit: Option<usize>,
    pub run: RunOptions,
    pub concept_search: Option<&'a dyn ConceptSearch>,
}

impl Default for RiverRequestOptions<'_> {
    fn default() -> Self {
        Self {
            observe: Default::default(),
            river: Default::default(),
            limit: Some(10),
            run: Default::default(),
            concept_search: None,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Baseline {
    OriginalCosine,
    #[default]
    TagMemo,
}

pub struct DtscRequestOptions<'a> {
    pub observe: ObserveConfig,
    pub dtsc: DtscConfig,
    pub baseline: Baseline,
    pub limit: Option<usize>,
    pub run: RunOptions,
    pub concept_search: Option<&'a dyn ConceptSearch>,
}

impl Default for DtscRequestOptions<'_> {
    fn default() -> Self {
        Self {
            observe: Default::default(),
            dtsc: Default::default(),
            baseline: Baseline::TagMemo,
            limit: Some(10),
            run: Default::default(),
            concept_search: None,
        }
    }
}

pub struct TagMemoRequestOptions<'a> {
    pub observe: ObserveConfig,
    pub limit: Option<usize>,
    pub run: RunOptions,
    pub concept_search: Option<&'a dyn ConceptSearch>,
}

impl Default for TagMemoRequestOptions<'_> {
    fn default() -> Self {
        Self {
            observe: Default::default(),
            limit: Some(10),
            run: Default::default(),
            concept_search: None,
        }
    }
}

impl Model {
    pub fn rank_river(
        &self,
        query: Query<'_>,
        candidates: &Candidates<'_>,
        options: &RiverRequestOptions<'_>,
    ) -> Result<Ranking> {
        let obs = self.observe_with(
            query,
            &options.observe,
            &options.run,
            options.concept_search,
        )?;
        let eval = obs.evaluate(
            candidates,
            &EvaluateOptions {
                run: options.run.clone(),
            },
        )?;
        let scores = eval.river(&RiverOptions {
            config: options.river.clone(),
            run: options.run.clone(),
        })?;
        let ranking = scores.top(options.limit);
        options.run.check_cancelled()?;
        Ok(ranking)
    }

    pub fn rank_dtsc(
        &self,
        query: Query<'_>,
        candidates: &Candidates<'_>,
        options: &DtscRequestOptions<'_>,
    ) -> Result<Ranking> {
        let obs = self.observe_with(
            query,
            &options.observe,
            &options.run,
            options.concept_search,
        )?;
        let eval = obs.evaluate(
            candidates,
            &EvaluateOptions {
                run: options.run.clone(),
            },
        )?;
        let base = match options.baseline {
            Baseline::OriginalCosine => eval.cosine(),
            Baseline::TagMemo => eval.tag_memo(),
        };
        let ranking = eval
            .dtsc(
                &base,
                &DtscOptions {
                    config: options.dtsc.clone(),
                    run: options.run.clone(),
                },
            )?
            .top(options.limit);
        options.run.check_cancelled()?;
        Ok(ranking)
    }

    pub fn rank_tag_memo(
        &self,
        query: Query<'_>,
        candidates: &Candidates<'_>,
        options: &TagMemoRequestOptions<'_>,
    ) -> Result<Ranking> {
        let obs = self.observe_with(
            query,
            &options.observe,
            &options.run,
            options.concept_search,
        )?;
        let eval = obs.evaluate(
            candidates,
            &EvaluateOptions {
                run: options.run.clone(),
            },
        )?;
        let ranking = eval.tag_memo().top(options.limit);
        options.run.check_cancelled()?;
        Ok(ranking)
    }
}

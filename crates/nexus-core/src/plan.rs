// SPDX-License-Identifier: Apache-2.0

use crate::{ExecutionContext, Operator, VcpError, VcpResult};
use std::any::Any;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicUsize, Ordering};

type Value = Box<dyn Any + Send + Sync>;
type Slots = [Option<Value>];
type Run<'request> = Box<dyn FnOnce(&ExecutionContext<'_>, &Slots) -> VcpResult<Value> + 'request>;

static NEXT_PLAN: AtomicUsize = AtomicUsize::new(1);

pub struct Handle<T> {
    plan: usize,
    slot: usize,
    value: PhantomData<fn() -> T>,
}

impl<T> Copy for Handle<T> {}

impl<T> Clone for Handle<T> {
    fn clone(&self) -> Self {
        *self
    }
}

fn invalid(detail: &str) -> VcpError {
    VcpError::InvalidInput {
        stage: "plan",
        detail: detail.into(),
    }
}

fn value<T: 'static>(slots: &Slots, slot: usize) -> VcpResult<&T> {
    slots
        .get(slot)
        .and_then(Option::as_ref)
        .and_then(|value| value.downcast_ref())
        .ok_or_else(|| invalid("result was not requested or is unavailable"))
}

mod dependencies {
    use super::*;

    pub trait Sealed: Copy {
        type Borrowed<'a>;
        fn collect(self, plan: usize, slots: &mut Vec<usize>) -> VcpResult<()>;
        fn borrow(self, slots: &Slots) -> VcpResult<Self::Borrowed<'_>>;
    }

    impl Sealed for () {
        type Borrowed<'a> = ();
        fn collect(self, _: usize, _: &mut Vec<usize>) -> VcpResult<()> {
            Ok(())
        }
        fn borrow(self, _: &Slots) -> VcpResult<()> {
            Ok(())
        }
    }

    impl<T: 'static> Sealed for Handle<T> {
        type Borrowed<'a> = &'a T;
        fn collect(self, plan: usize, slots: &mut Vec<usize>) -> VcpResult<()> {
            if self.plan != plan {
                return Err(invalid("handle belongs to another plan"));
            }
            slots.push(self.slot);
            Ok(())
        }
        fn borrow(self, slots: &Slots) -> VcpResult<&T> {
            value(slots, self.slot)
        }
    }

    macro_rules! tuple {
        ($($ty:ident : $i:tt),+) => {
            impl<$($ty: Sealed),+> Sealed for ($($ty,)+) {
                type Borrowed<'a> = ($($ty::Borrowed<'a>,)+);
                fn collect(self, plan: usize, slots: &mut Vec<usize>) -> VcpResult<()> {
                    $(self.$i.collect(plan, slots)?;)+
                    Ok(())
                }
                fn borrow(self, slots: &Slots) -> VcpResult<Self::Borrowed<'_>> {
                    Ok(($(self.$i.borrow(slots)?,)+))
                }
            }
        };
    }

    tuple!(A:0, B:1);
    tuple!(A:0, B:1, C:2);
}

/// A handle, tuple of handles, or the empty dependency list.
pub trait Dependencies: dependencies::Sealed {}
impl<T: dependencies::Sealed> Dependencies for T {}

/// The requested dependencies, already evaluated and type-checked by the plan.
pub struct Inputs<'result, D: Dependencies> {
    values: D::Borrowed<'result>,
}

impl<'result, D: Dependencies> Inputs<'result, D> {
    pub fn get(self) -> D::Borrowed<'result> {
        self.values
    }
}

struct Step<'request> {
    inputs: Vec<usize>,
    run: Run<'request>,
}

/// An in-memory composition for one request. Handles point to earlier operations.
pub struct Plan<'request> {
    id: usize,
    steps: Vec<Step<'request>>,
}

impl Default for Plan<'_> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'request> Plan<'request> {
    pub fn new() -> Self {
        Self {
            id: NEXT_PLAN.fetch_add(1, Ordering::Relaxed),
            steps: Vec::new(),
        }
    }

    /// The binder receives the typed results of the requested dependencies.
    pub fn add<O, D, F>(
        &mut self,
        operator: O,
        config: O::Config,
        dependencies: D,
        bind: F,
    ) -> VcpResult<Handle<O::Output>>
    where
        O: Operator + 'static,
        O::Config: 'request,
        D: Dependencies + 'request,
        F: for<'results> FnOnce(Inputs<'results, D>) -> VcpResult<O::Input<'request, 'results>>
            + 'request,
    {
        let mut inputs = Vec::new();
        dependencies.collect(self.id, &mut inputs)?;
        let slot = self.steps.len();
        self.steps.push(Step {
            inputs,
            run: Box::new(move |ctx, slots| {
                ctx.check_cancelled(operator.name())?;
                let input = bind(Inputs {
                    values: dependencies.borrow(slots)?,
                })?;
                Ok(Box::new(operator.execute(ctx, &config, input)?))
            }),
        });
        Ok(Handle {
            plan: self.id,
            slot,
            value: PhantomData,
        })
    }

    pub fn run(
        self,
        outputs: impl Dependencies,
        ctx: &ExecutionContext<'_>,
    ) -> VcpResult<PlanResults> {
        let mut requested = Vec::new();
        outputs.collect(self.id, &mut requested)?;
        let mut uses = vec![0_usize; self.steps.len()];
        for output in requested {
            uses[output] += 1;
        }
        for (slot, step) in self.steps.iter().enumerate().rev() {
            if uses[slot] > 0 {
                for &input in &step.inputs {
                    uses[input] += 1;
                }
            }
        }
        let mut values: Vec<Option<Value>> = (0..uses.len()).map(|_| None).collect();
        for (slot, step) in self.steps.into_iter().enumerate() {
            if uses[slot] == 0 {
                continue;
            }
            values[slot] = Some((step.run)(ctx, &values)?);
            for input in step.inputs {
                uses[input] -= 1;
                if uses[input] == 0 {
                    values[input] = None;
                }
            }
        }
        Ok(PlanResults {
            id: self.id,
            values,
        })
    }
}

pub struct PlanResults {
    id: usize,
    values: Vec<Option<Value>>,
}

impl PlanResults {
    pub fn get<T: 'static>(&self, result: Handle<T>) -> VcpResult<&T> {
        if result.plan != self.id {
            return Err(invalid("handle belongs to another plan"));
        }
        value(&self.values, result.slot)
    }
}

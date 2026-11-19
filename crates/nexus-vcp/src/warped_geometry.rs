// SPDX-License-Identifier: Apache-2.0
// Numerical geometry from VCPToolBox 5adb12b, with original attribution below.
//! Adapted from Pelorus 98136b2, Apache-2.0.
//! Copyright 2024 TriviumDB Authors (https://github.com/YoKONCy).
//! Numerical reference and bounded-domain interpolation, NOT conservative bounds.

/// 球对称翘曲乘积流形径向缩放剖面 φ(R)。
///
/// Original profiles have a regular collapsed origin. ThroatSoftening instead
/// defines a manifold with boundary when phi0>0, or a metric cone when kappa!=1.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum WarpedProfile {
    /// 常负曲率特例 φ(R) = sinh(sqrt(c)*R)/sqrt(c)，专用于验证 Clairaut 数值求积与闭式解一致。
    ConstantHyperbolic { curvature: f64 },
    /// 外层逐渐变平（核心强双曲、外围渐平）：φ(R) = R + α R^3，K_rad(R) = -6α / (1 + α R^2)
    Softening { alpha: f64 },
    /// 外层曲率逐渐变陡（双曲喇叭型）：φ(R) = R exp(α R^2)，K_rad(R) = -(6α + 4α^2 R^2)
    Steepening { alpha: f64 },
    /// Intrinsic length metric on [0,infinity) x S^(D-1), including boundary
    /// arcs at R=0 when phi0>0. If phi0=0 the boundary sphere collapses.
    ThroatSoftening { alpha: f64, phi0: f64, kappa: f64 },
}

impl WarpedProfile {
    #[inline]
    pub fn phi(self, r: f64) -> f64 {
        match self {
            Self::ConstantHyperbolic { curvature } => {
                let s = curvature.sqrt();
                (s * r).sinh() / s
            }
            Self::Softening { alpha } => r + alpha * r * r * r,
            Self::Steepening { alpha } => r * (alpha * r * r).exp(),
            Self::ThroatSoftening { alpha, phi0, kappa } => kappa * (phi0 + r + alpha * r * r * r),
        }
    }

    #[inline]
    pub fn radial_curvature(self, r: f64) -> f64 {
        match self {
            Self::ConstantHyperbolic { curvature } => -curvature,
            Self::Softening { alpha } => -6.0 * alpha / (1.0 + alpha * r * r),
            Self::Steepening { alpha } => -(6.0 * alpha + 4.0 * alpha * alpha * r * r),
            Self::ThroatSoftening { alpha, phi0, .. } => {
                if r == 0.0 && phi0 == 0.0 {
                    -6.0 * alpha
                } else {
                    -6.0 * alpha * r / (phi0 + r + alpha * r * r * r)
                }
            }
        }
    }
}

/// 16 点 Gauss-Legendre 求积节点与权重（在 [0, 1] 上的对称半区间缩放至 [-1, 1]）。
const GL_X: [f64; 8] = [
    0.09501250983763744,
    0.2816035507792589,
    0.4580167776572274,
    0.6178762444026438,
    0.755404408355003,
    0.8656312023878318,
    0.9445750230732326,
    0.9894009349916499,
];
const GL_W: [f64; 8] = [
    0.1894506104550685,
    0.18260341504492358,
    0.16915651939500254,
    0.14959598881657674,
    0.12462897125553388,
    0.09515851168249279,
    0.062253523938647894,
    0.027152459411754096,
];

const PANELS: [(f64, f64); 8] = [
    (0.0, 0.01),
    (0.01, 0.04),
    (0.04, 0.09),
    (0.09, 0.16),
    (0.16, 0.30),
    (0.30, 0.45),
    (0.45, 0.70),
    (0.70, 1.0),
];
const QUAD_POINTS: usize = PANELS.len() * 16;

const fn build_clairaut_quad_table() -> ([f64; QUAD_POINTS], [f64; QUAD_POINTS]) {
    let mut u2 = [0.0f64; QUAD_POINTS];
    let mut wu2 = [0.0f64; QUAD_POINTS];
    let mut pi = 0usize;
    while pi < PANELS.len() {
        let (u0, u1) = PANELS[pi];
        let mid = 0.5 * (u0 + u1);
        let half = 0.5 * (u1 - u0);
        let mut k = 0usize;
        while k < 8 {
            let w = GL_W[k] * half;
            let dx = half * GL_X[k];
            let ua = mid - dx;
            let ub = mid + dx;
            let idx0 = pi * 16 + k * 2;
            u2[idx0] = ua * ua;
            wu2[idx0] = 2.0 * w * ua;
            u2[idx0 + 1] = ub * ub;
            wu2[idx0 + 1] = 2.0 * w * ub;
            k += 1;
        }
        pi += 1;
    }
    (u2, wu2)
}

const CLAIRAUT_QUAD: ([f64; QUAD_POINTS], [f64; QUAD_POINTS]) = build_clairaut_quad_table();

#[inline(always)]
fn clairaut_quad<F>(profile: WarpedProfile, a: f64, b: f64, ang_mom: f64, integrand: F) -> f64
where
    F: Fn(f64, f64) -> f64,
{
    let span = b - a;
    if span <= 0.0 {
        return 0.0;
    }
    let pa = profile.phi(a);
    let (u2_tbl, wu2_tbl) = &CLAIRAUT_QUAD;
    let mut sum = 0.0f64;
    for i in 0..QUAD_POINTS {
        let delta = span * u2_tbl[i];
        let r = a + delta;
        let p = profile.phi(r);
        let increment = match profile {
            WarpedProfile::ConstantHyperbolic { curvature } => {
                let s = curvature.sqrt();
                2.0 * (s * delta * 0.5).sinh() * (s * (r + a) * 0.5).cosh() / s
            }
            WarpedProfile::Softening { alpha } => delta * (1.0 + alpha * (r * r + r * a + a * a)),
            WarpedProfile::Steepening { alpha } => {
                (alpha * a * a).exp() * (delta + r * (alpha * delta * (r + a)).exp_m1())
            }
            WarpedProfile::ThroatSoftening { alpha, kappa, .. } => {
                kappa * delta * (1.0 + alpha * (r * r + r * a + a * a))
            }
        };
        let root = ((increment + (pa - ang_mom)).max(0.0) * (p + ang_mom))
            .max(1e-30)
            .sqrt();
        sum += wu2_tbl[i] * integrand(p, root);
    }
    sum * span
}

/// 仅计算角度积分项 Δθ（供 Brent 求根迭代内部调用，省去 30% 弧长浮点开销）。
#[inline]
fn clairaut_theta(profile: WarpedProfile, a: f64, b: f64, ang_mom: f64) -> f64 {
    clairaut_quad(profile, a, b, ang_mom, |p, root| 1.0 / (p * root)) * ang_mom
}

/// 仅计算测地弧长积分项（仅在根收敛后计算一次）。
#[inline]
fn clairaut_length(profile: WarpedProfile, a: f64, b: f64, ang_mom: f64) -> f64 {
    clairaut_quad(profile, a, b, ang_mom, |p, root| p / root)
}

/// 分段标量 Brent / 保护二分求根器：求解单调函数 f(x) = target 在 [lo, hi] 上的根。
fn solve_monotone<F>(lo: f64, hi: f64, f_lo: f64, f_hi: f64, target: f64, f: F) -> f64
where
    F: Fn(f64) -> f64,
{
    let mut a = lo;
    let mut b = hi;
    let mut fa = f_lo - target;
    let mut fb = f_hi - target;
    if fa.abs() < 1e-12 {
        return a;
    }
    if fb.abs() < 1e-12 {
        return b;
    }
    if fa.signum() == fb.signum() {
        return if fa.abs() <= fb.abs() { a } else { b };
    }
    let mut c = a;
    let mut fc = fa;
    let mut d = b - a;
    let mut e = d;
    for _ in 0..48 {
        if fb == 0.0 {
            break;
        }
        if fc.signum() == fb.signum() {
            c = a;
            fc = fa;
            d = b - a;
            e = d;
        }
        if fc.abs() < fb.abs() {
            a = b;
            b = c;
            c = a;
            fa = fb;
            fb = fc;
            fc = fa;
        }
        // Root tolerance; not a global quadrature-error guarantee.
        let tol = 2.0 * 1e-12 * b.abs().max(1.0);
        let m = 0.5 * (c - b);
        if m.abs() <= tol || fb == 0.0 {
            break;
        }
        if e.abs() >= tol && fa.abs() > fb.abs() {
            let s = fb / fa;
            let (mut p, mut q) = if (a - c).abs() < 1e-15 {
                (2.0 * m * s, 1.0 - s)
            } else {
                let q0 = fa / fc;
                let r = fb / fc;
                (
                    s * (2.0 * m * q0 * (q0 - r) - (b - a) * (r - 1.0)),
                    (q0 - 1.0) * (r - 1.0) * (s - 1.0),
                )
            };
            if p > 0.0 {
                q = -q;
            } else {
                p = -p;
            }
            if 2.0 * p < (3.0 * m * q - (tol * q).abs()).min((e * q).abs()) {
                e = d;
                d = p / q;
            } else {
                d = m;
                e = m;
            }
        } else {
            d = m;
            e = m;
        }
        a = b;
        fa = fb;
        b += if d.abs() > tol { d } else { tol.copysign(m) };
        fb = f(b) - target;
    }
    b
}

/// 利用 Clairaut 守恒积分计算径向变曲率翘曲乘积流形的精确黎曼测地距离。
pub fn clairaut_distance(profile: WarpedProfile, rq: f64, rv: f64, cos_theta: f64) -> f64 {
    if matches!(profile, WarpedProfile::ThroatSoftening { .. }) {
        return throat_distance(profile, rq, rv, cos_theta);
    }
    if rq <= 1e-12 {
        return rv.max(0.0);
    }
    if rv <= 1e-12 {
        return rq.max(0.0);
    }
    let cos_c = cos_theta.clamp(-1.0, 1.0);
    if cos_c >= 1.0 - 1e-14 {
        return (rq - rv).abs();
    }
    if cos_c <= -1.0 + 1e-14 {
        return rq + rv;
    }
    let theta = cos_c.acos();
    let (r_lo, r_hi) = if rq <= rv { (rq, rv) } else { (rv, rq) };
    let l_max = profile.phi(r_lo);

    // 单调分支最大角度（当 r_lo < r_hi 时从 r_lo 出发直达 r_hi）
    let theta_crit = if r_hi > r_lo + 1e-13 {
        clairaut_theta(profile, r_lo, r_hi, l_max)
    } else {
        0.0
    };

    if theta <= theta_crit && r_hi > r_lo + 1e-13 {
        // Case 1: 径向单调上升段 L ∈ [0, phi(r_lo)]
        let eval_theta = |ang_mom: f64| clairaut_theta(profile, r_lo, r_hi, ang_mom);
        let l_star = solve_monotone(0.0, l_max, 0.0, theta_crit, theta, eval_theta);
        return clairaut_length(profile, r_lo, r_hi, l_star);
    }

    // Case 2: 测地线先向内折返到近日点 r_* ∈ [0, r_lo]，再向外爬升经过 r_lo 至 r_hi
    if (theta - theta_crit).abs() <= 1e-14 {
        return clairaut_length(profile, r_lo, r_hi, l_max);
    }
    let eval_dip_theta = |r_turn: f64| -> f64 {
        if r_turn <= 1e-13 {
            return std::f64::consts::PI;
        }
        let l = profile.phi(r_turn);
        // Both legs start at the turning point, so the squared substitution
        // regularizes both endpoint singularities, including near-critical dips.
        clairaut_theta(profile, r_turn, r_lo, l) + clairaut_theta(profile, r_turn, r_hi, l)
    };
    let r_star = solve_monotone(
        0.0,
        r_lo,
        std::f64::consts::PI,
        theta_crit,
        theta,
        eval_dip_theta,
    );
    let l = profile.phi(r_star);
    clairaut_length(profile, r_star, r_lo, l) + clairaut_length(profile, r_star, r_hi, l)
}

/// Numerical length metric with an explicit reflecting obstacle boundary.
/// Positive throat: tangent legs may be joined by a nonzero boundary arc.
/// Zero throat and kappa!=1: a metric cone, NOT a smooth Euclidean origin.
fn throat_distance(profile: WarpedProfile, rq: f64, rv: f64, cosine: f64) -> f64 {
    let theta = cosine.clamp(-1.0, 1.0).acos();
    if theta == 0.0 {
        return (rq - rv).abs();
    }
    let (lo, hi) = (rq.min(rv), rq.max(rv));
    let base = profile.phi(0.0);
    if base == 0.0 && lo == 0.0 {
        return hi;
    }
    let lmax = profile.phi(lo);
    let critical = clairaut_theta(profile, lo, hi, lmax);
    if theta <= critical && hi > lo {
        let l = solve_monotone(0.0, lmax, 0.0, critical, theta, |l| {
            clairaut_theta(profile, lo, hi, l)
        });
        return clairaut_length(profile, lo, hi, l);
    }
    let maximum = {
        if base > 0.0 {
            clairaut_theta(profile, 0.0, lo, base) + clairaut_theta(profile, 0.0, hi, base)
        } else {
            match profile {
                WarpedProfile::ThroatSoftening { kappa, .. } => std::f64::consts::PI / kappa,
                _ => unreachable!(),
            }
        }
    };
    if theta >= maximum {
        if base == 0.0 {
            return rq + rv;
        }
        let leg_length_sum =
            clairaut_length(profile, 0.0, lo, base) + clairaut_length(profile, 0.0, hi, base);
        return leg_length_sum + base * (theta - maximum);
    }
    let turn = solve_monotone(0.0, lo, maximum, critical, theta, |r| {
        if r == 0.0 {
            return maximum;
        }
        let l = profile.phi(r);
        clairaut_theta(profile, r, lo, l) + clairaut_theta(profile, r, hi, l)
    });
    let l = profile.phi(turn);
    clairaut_length(profile, turn, lo, l) + clairaut_length(profile, turn, hi, l)
}

/// Metric comparison: nonnegative cubic softening dominates the linear throat.
/// This is not an interpolated-table envelope. The floating-point retreat is
/// conservative engineering protection, not certified quadrature error analysis.
pub fn clairaut_lower_bound(profile: WarpedProfile, rq: f64, rv: f64, cosine: f64) -> f64 {
    if !rq.is_finite() || !rv.is_finite() || rq < 0.0 || rv < 0.0 || !cosine.is_finite() {
        return 0.0;
    }
    // Legacy Brent endpoint tolerance can collapse extremely near-aligned
    // equal-radius distances to zero. Disable pruning in that unresolved region
    // rather than changing the historical numerical target.
    if cosine >= 1.0 - 1e-8 {
        return 0.0;
    }
    let radial = (rq - rv).abs();
    let raw = match profile {
        WarpedProfile::ThroatSoftening { alpha, phi0, kappa }
            if alpha >= 0.0 && phi0 >= 0.0 && kappa > 0.0 =>
        {
            let theta = cosine.clamp(-1.0, 1.0).acos();
            let angle = kappa * theta;
            let (x, y) = (rq + phi0, rv + phi0);
            let critical = if phi0 == 0.0 {
                std::f64::consts::PI
            } else {
                (phi0 / x).clamp(0.0, 1.0).acos() + (phi0 / y).clamp(0.0, 1.0).acos()
            };
            let linear = if angle <= critical {
                (radial * radial + 4.0 * x * y * (angle * 0.5).sin().powi(2)).sqrt()
            } else {
                ((x - phi0) * (x + phi0)).max(0.0).sqrt()
                    + ((y - phi0) * (y + phi0)).max(0.0).sqrt()
                    + phi0 * (angle - critical)
            };
            let cylinder = (radial * radial + (kappa * phi0 * theta).powi(2)).sqrt();
            linear.max(cylinder)
        }
        _ => radial,
    };
    let path_scale = 1.0 + raw.abs() + rq + rv + profile.phi(rq).abs() + profile.phi(rv).abs();
    let machine_guard = 128.0 * f64::EPSILON * path_scale;
    // Keep the historical quadrature retreat as well as the scale-aware
    // arithmetic guard: machine epsilon alone does not bound Brent/quadrature.
    let numerical_guard = 1e-7 * (1.0 + raw + rq + rv);
    if raw.is_finite() && machine_guard.is_finite() {
        (raw - numerical_guard - machine_guard).max(0.0)
    } else {
        0.0
    }
}

"""Analyze a completed A/B test.  ⛏️  EXERCISE — Milestone 5.

LEARNING GOAL
    Given control/treatment outcomes, produce the full answer an interviewer wants:
    not just a p-value, but the effect size, a confidence interval, and a guardrail
    check. "p < 0.05" alone is a junior answer; "+3.2% relative lift, 95% CI
    [1.1%, 5.3%], p=0.004, no guardrail regression" is a senior one.

KEY CONCEPT — two-proportion z-test + CI on the difference
    For conversion rates: pooled-proportion z-test for the p-value; for the CI use
    the UNpooled standard error of (p_t - p_c):
        SE = sqrt( p_c(1-p_c)/n_c + p_t(1-p_t)/n_t )
        CI = (p_t - p_c) ± z_{1-α/2} * SE
    Report absolute diff, relative lift, the CI, and the p-value.

INTERVIEW ANGLE
    Always pair the primary metric with GUARDRAILS (the doc: latency, return rate,
    satisfaction). "Conversion improved but returns also rose" is a real possible
    outcome — surface it, don't bury it.

TEST
    tests/test_ab_analyze.py
"""

from __future__ import annotations

from dataclasses import dataclass

from scipy.stats import norm


@dataclass
class ABResult:
    control_rate: float
    treatment_rate: float
    absolute_diff: float
    relative_lift: float
    ci_low: float
    ci_high: float
    p_value: float
    significant: bool


def two_proportion_test(
    control_conversions: int,
    control_n: int,
    treatment_conversions: int,
    treatment_n: int,
    alpha: float = 0.05,
) -> ABResult:
    """Run a two-proportion z-test and return effect size + CI + significance.

    TODO(lillian): implement per KEY CONCEPT.
      - p_c, p_t = rates; absolute_diff = p_t - p_c; relative_lift = diff / p_c
      - pooled p for the test statistic; unpooled SE for the CI
      - z = diff / SE_pooled ; p_value = 2 * (1 - norm.cdf(|z|))  (two-sided)
      - CI = diff ± norm.ppf(1-alpha/2) * SE_unpooled
      - significant = p_value < alpha
    The test cross-checks against statsmodels.proportions_ztest on a known case.
    """
    p_c, p_t = control_conversions / control_n, treatment_conversions / treatment_n
    absolute_diff = p_t - p_c
    relative_lift = absolute_diff / p_c if p_c != 0 else 0
    p_pooled = (control_conversions + treatment_conversions) / (control_n + treatment_n)
    se_pooled = (p_pooled * (1 - p_pooled) * ( 1 / control_n + 1 / treatment_n)) ** 0.5
    z = absolute_diff / se_pooled
    p_value = 2 * (1 - norm.cdf(abs(z)))
    se_unpooled = ((p_c * (1 - p_c) / control_n) + (p_t * (1 - p_t) / treatment_n)) ** 0.5
    ci_low = absolute_diff - norm.ppf(1 - alpha / 2) * se_unpooled
    ci_high = absolute_diff + norm.ppf(1 - alpha / 2) * se_unpooled
    significant = bool(p_value < alpha)
    return ABResult(
        control_rate=p_c,
        treatment_rate=p_t,
        absolute_diff=absolute_diff,
        relative_lift=relative_lift,
        ci_low=ci_low,
        ci_high=ci_high,
        p_value=p_value,
        significant=significant
    )
    # raise NotImplementedError("Implement two_proportion_test — see tests/test_ab_analyze.py")


def check_guardrails(results: dict[str, ABResult], guardrail_metrics: list[str]) -> dict[str, bool]:
    """Return {guardrail: passed?} — a guardrail FAILS if treatment significantly
    worsened it.

    TODO(lillian): for each guardrail metric's ABResult, decide pass/fail. You must
    define 'worse' per metric (higher return-rate is bad; higher satisfaction is
    good) — encode that direction explicitly rather than assuming. Document the
    convention you choose.
    """
    # raise NotImplementedError
    HIGHER_IS_WORSE = {"return_rate"}
    passed_guardrails = {}
    for metric in guardrail_metrics:
        result = results[metric]
        if not result.significant:
            passed_guardrails[metric] = True  # not significant → pass
        
        elif metric in HIGHER_IS_WORSE:
            passed_guardrails[metric] = result.relative_lift <= 0  # higher is worse → pass if lift ≤ 0
        else:
            passed_guardrails[metric] = result.relative_lift >= 0  # higher is better → pass if lift ≥ 0
    return passed_guardrails

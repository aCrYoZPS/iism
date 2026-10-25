use crate::task::{Param, Solution, Task, require_positive};

const STAGES: usize = 3;
const MINUTES_PER_HOUR: f64 = 60.0;

/// Metric names per stage; `Metric` names are static, so they are spelled out.
const STAGE_METRICS: [[&str; 5]; STAGES] = [
    [
        "1 Choice: ρ",
        "1 Choice: P(wait)",
        "1 Choice: mean queue",
        "1 Choice: wait, min",
        "1 Choice: time in stage, min",
    ],
    [
        "2 Cashier: ρ",
        "2 Cashier: P(wait)",
        "2 Cashier: mean queue",
        "2 Cashier: wait, min",
        "2 Cashier: time in stage, min",
    ],
    [
        "3 Control: ρ",
        "3 Control: P(wait)",
        "3 Control: mean queue",
        "3 Control: wait, min",
        "3 Control: time in stage, min",
    ],
];

/// Shop as a series of three multi-channel queues. By Burke's theorem the
/// output of a stationary M/M/n queue is again a Poisson flow of rate λ, so
/// every stage is solved on its own with the Erlang C formulas.
pub fn task() -> Task {
    Task {
        name: "Shop (series of queues)",
        params: vec![
            Param::real("λ, per hour", 45.0),
            Param::count("n1", 4, 1),
            Param::real("t1, min", 5.0),
            Param::count("n2", 1, 1),
            Param::real("t2, min", 1.0),
            Param::count("n3", 3, 1),
            Param::real("t3, min", 1.0),
        ],
        solve,
    }
}

struct Stage {
    rho: f64,
    wait_probability: f64,
    queue: f64,
    wait: f64,
    time_in_stage: f64,
}

/// M/M/n with an unbounded queue; rates per hour, times returned in hours.
fn erlang_c(lambda: f64, mu: f64, channels: usize) -> Result<Stage, String> {
    let load = lambda / mu;
    let rho = load / channels as f64;
    if rho >= 1.0 {
        return Err(format!(
            "ρ = {rho:.4} ≥ 1: the queue grows without limit, no stationary regime"
        ));
    }

    // sum_{k<n} a^k / k!, with the term a^n / n! left over at the end
    let mut term = 1.0;
    let mut below = 0.0;
    for k in 0..channels {
        below += term;
        term *= load / (k + 1) as f64;
    }
    let at_capacity = term / (1.0 - rho);
    let wait_probability = at_capacity / (below + at_capacity);

    let queue = wait_probability * rho / (1.0 - rho);
    let wait = queue / lambda;
    Ok(Stage {
        rho,
        wait_probability,
        queue,
        wait,
        time_in_stage: wait + 1.0 / mu,
    })
}

fn solve(params: &[f64]) -> Result<Solution, String> {
    let &[lambda, n1, t1, n2, t2, n3, t3] = params else {
        return Err("wrong number of parameters".into());
    };
    require_positive("λ", lambda)?;
    let setup = [(n1, t1, "t1"), (n2, t2, "t2"), (n3, t3, "t3")];

    let mut stages = Vec::with_capacity(STAGES);
    for (i, &(channels, minutes, name)) in setup.iter().enumerate() {
        require_positive(name, minutes)?;
        let mu = MINUTES_PER_HOUR / minutes;
        let stage = erlang_c(lambda, mu, channels as usize)
            .map_err(|message| format!("stage {}: {message}", i + 1))?;
        stages.push(stage);
    }

    let time_in_shop: f64 = stages.iter().map(|s| s.time_in_stage).sum();
    let bottleneck = stages
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.rho.total_cmp(&b.1.rho))
        .map_or(0, |(i, _)| i);

    let mut solution = Solution::new(Vec::new(), Vec::new());
    for (names, stage) in STAGE_METRICS.iter().zip(&stages) {
        let values = [
            stage.rho,
            stage.wait_probability,
            stage.queue,
            stage.wait * MINUTES_PER_HOUR,
            stage.time_in_stage * MINUTES_PER_HOUR,
        ];
        for (&name, value) in names.iter().zip(values) {
            solution = solution.metric(name, value);
        }
    }
    Ok(solution
        .metric("Time in shop, min", time_in_shop * MINUTES_PER_HOUR)
        .metric("Customers in shop", lambda * time_in_shop)
        .metric("A (absolute throughput)", lambda)
        .metric("Q (relative throughput)", 1.0)
        .metric("P_reject", 0.0)
        .metric("Bottleneck stage (max ρ)", (bottleneck + 1) as f64))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tasks::queue;

    #[test]
    fn single_channel_matches_mm1_formulas() {
        // M/M/1: Lq = rho^2 / (1 - rho), W = 1 / (mu - lambda)
        let (lambda, mu) = (45.0, 60.0);
        let rho = lambda / mu;

        let stage = erlang_c(lambda, mu, 1).unwrap();

        assert!((stage.queue - rho * rho / (1.0 - rho)).abs() < 1e-12);
        assert!((stage.time_in_stage - 1.0 / (mu - lambda)).abs() < 1e-12);
        assert!((stage.wait_probability - rho).abs() < 1e-12);
    }

    #[test]
    fn multi_channel_matches_the_queue_tab() {
        let (lambda, mu, channels) = (1.7, 0.6, 4);

        let stage = erlang_c(lambda, mu, channels).unwrap();
        let plain =
            (queue::task().solve)(&[lambda, mu, channels as f64, 0.0, 1.0])
                .unwrap();

        let queue_len = plain.find("Mean queue length").unwrap();
        let time = plain.find("Mean time in system").unwrap();
        assert!((stage.queue - queue_len).abs() < 1e-6, "{}", stage.queue);
        assert!((stage.time_in_stage - time).abs() < 1e-6);
    }

    #[test]
    fn shop_totals_add_up_and_the_busiest_stage_is_the_bottleneck() {
        let solution = solve(&[45.0, 4.0, 5.0, 1.0, 1.0, 3.0, 1.0]).unwrap();

        let stage_times: f64 = [
            "1 Choice: time in stage, min",
            "2 Cashier: time in stage, min",
            "3 Control: time in stage, min",
        ]
        .iter()
        .map(|name| solution.find(name).unwrap())
        .sum();
        assert!(
            (solution.find("Time in shop, min").unwrap() - stage_times).abs()
                < 1e-9
        );
        // Little's law for the whole shop
        let customers = solution.find("Customers in shop").unwrap();
        assert!(
            (customers - 45.0 * stage_times / MINUTES_PER_HOUR).abs() < 1e-9
        );
        assert_eq!(solution.find("Bottleneck stage (max ρ)"), Some(1.0));
        assert!((solution.find("1 Choice: ρ").unwrap() - 0.9375).abs() < 1e-12);
        assert!((solution.find("2 Cashier: ρ").unwrap() - 0.75).abs() < 1e-12);
    }

    #[test]
    fn overloaded_stage_is_rejected_with_its_number() {
        // 3 controllers, 5 min each: rho = 45 / 36 > 1
        let error = solve(&[45.0, 4.0, 5.0, 1.0, 1.0, 3.0, 5.0]).err().unwrap();

        assert!(error.starts_with("stage 3"), "{error}");
    }
}

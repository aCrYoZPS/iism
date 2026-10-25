use crate::{
    chain::Chain,
    task::{Param, Solution, Task, require_positive},
};

/// States kept when the queue is unbounded; the tail beyond is negligible
/// while lambda < n * mu is not close to equality.
const TRUNCATION: usize = 200;
const TAIL_LIMIT: f64 = 1e-9;

/// n channels, m queue places or an unbounded queue (tasks 1, 3, 4-5, ...).
/// State k is the number of requests in the system.
pub fn task() -> Task {
    Task {
        name: "Channels and queue",
        params: vec![
            Param::real("λ", 0.4),
            Param::real("μ", 1.0 / 3.0),
            Param::count("n", 1, 1),
            Param::count("m", 0, 0),
            Param::flag("unbounded queue", false),
        ],
        solve,
    }
}

fn solve(params: &[f64]) -> Result<Solution, String> {
    let &[lambda, mu, n, m, unbounded] = params else {
        return Err("wrong number of parameters".into());
    };
    require_positive("λ", lambda)?;
    require_positive("μ", mu)?;
    let (n, m, unbounded) = (n as usize, m as usize, unbounded > 0.5);

    if unbounded && lambda >= n as f64 * mu {
        return Err(format!(
            "λ ≥ nμ: the queue grows without limit, no stationary regime (ρ = {:.4})",
            lambda / (n as f64 * mu)
        ));
    }

    let last = if unbounded { TRUNCATION } else { n + m };
    let mut chain = Chain::new(last + 1);
    for k in 0..last {
        chain.add_rate(k, k + 1, lambda);
        chain.add_rate(k + 1, k, (k + 1).min(n) as f64 * mu);
    }
    let p = chain
        .stationary()
        .ok_or("the chain has no unique solution")?;

    let expect = |weight: &dyn Fn(usize) -> f64| -> f64 {
        p.iter().enumerate().map(|(k, pk)| weight(k) * pk).sum()
    };
    let in_system = expect(&|k| k as f64);
    let in_queue = expect(&|k| k.saturating_sub(n) as f64);
    let busy = expect(&|k| k.min(n) as f64);

    let reject = if unbounded { 0.0 } else { p[last] };
    let throughput = lambda * (1.0 - reject);

    let tail = p[last];
    let labels = (0..=last).map(|k| k.to_string()).collect();
    let mut solution = Solution::new(labels, p.clone())
        .metric("P_reject", reject)
        .metric("Q (relative throughput)", 1.0 - reject)
        .metric("A (absolute throughput)", throughput)
        .metric("Mean requests in system", in_system)
        .metric("Mean queue length", in_queue)
        .metric("Mean busy channels", busy)
        .metric("Mean time in system", in_system / throughput)
        .metric("Mean time in queue", in_queue / throughput);

    if unbounded && tail > TAIL_LIMIT {
        solution = solution.warning(format!(
            "the queue was cut at {TRUNCATION} states and p_{TRUNCATION} = {tail:.2e}: results are inaccurate"
        ));
    }
    Ok(solution)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    #[test]
    fn task_1_single_channel_with_refusals() {
        let solution = solve(&[0.4, 1.0 / 3.0, 1.0, 0.0, 0.0]).unwrap();

        assert!(close(solution.p[0], 5.0 / 11.0));
        assert!(close(solution.find("P_reject").unwrap(), 6.0 / 11.0));
        assert!(close(
            solution.find("A (absolute throughput)").unwrap(),
            2.0 / 11.0
        ));
    }

    #[test]
    fn task_3_unbounded_queue_matches_the_formulas() {
        // lambda = 2 per hour, mu = 3 per hour
        let solution = solve(&[2.0, 3.0, 1.0, 0.0, 1.0]).unwrap();

        assert!(close(solution.p[0], 1.0 / 3.0));
        assert!(close(
            solution.find("Mean requests in system").unwrap(),
            2.0
        ));
        assert!(close(
            solution.find("Mean queue length").unwrap(),
            4.0 / 3.0
        ));
        assert!(close(solution.find("Mean time in system").unwrap(), 1.0));
        assert!(close(
            solution.find("Mean time in queue").unwrap(),
            2.0 / 3.0
        ));
    }

    #[test]
    fn overloaded_unbounded_queue_is_rejected() {
        assert!(solve(&[3.0, 3.0, 1.0, 0.0, 1.0]).is_err());
    }

    #[test]
    fn two_channels_without_queue_match_erlang_b() {
        // rho = 3.2, B = (rho^2/2) / (1 + rho + rho^2/2)
        let rho = 3.2_f64;
        let expected = (rho * rho / 2.0) / (1.0 + rho + rho * rho / 2.0);

        let solution = solve(&[4.0, 1.25, 2.0, 0.0, 0.0]).unwrap();

        assert!(close(solution.find("P_reject").unwrap(), expected));
    }
}

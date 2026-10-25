use crate::{
    chain::Chain,
    task::{Param, Solution, Task, require_positive},
};

/// One operator repairs n machines. State k is the number of broken machines,
/// so the failure rate falls as k grows.
pub fn task() -> Task {
    Task {
        name: "Machines and repairman",
        params: vec![
            Param::count("n", 5, 1),
            Param::real("λ (one machine)", 0.2),
            Param::real("μ", 1.0),
        ],
        solve,
    }
}

fn solve(params: &[f64]) -> Result<Solution, String> {
    let &[n, lambda, mu] = params else {
        return Err("wrong number of parameters".into());
    };
    require_positive("λ", lambda)?;
    require_positive("μ", mu)?;
    let n = n as usize;

    let mut chain = Chain::new(n + 1);
    for k in 0..n {
        chain.add_rate(k, k + 1, (n - k) as f64 * lambda);
        chain.add_rate(k + 1, k, mu);
    }
    let p = chain
        .stationary()
        .ok_or("the chain has no unique solution")?;

    let broken: f64 = p.iter().enumerate().map(|(k, pk)| k as f64 * pk).sum();
    let waiting: f64 = p
        .iter()
        .enumerate()
        .map(|(k, pk)| k.saturating_sub(1) as f64 * pk)
        .sum();
    let busy = 1.0 - p[0];

    let labels = (0..=n).map(|k| k.to_string()).collect();
    Ok(Solution::new(labels, p)
        .metric("A (machines repaired per unit time)", mu * busy)
        .metric("w (mean broken machines)", broken)
        .metric("R (mean machines waiting)", waiting)
        .metric("P_busy (repairman busy)", busy))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_machine_alternates_between_two_states() {
        let solution = solve(&[1.0, 2.0, 3.0]).unwrap();

        assert!((solution.p[0] - 3.0 / 5.0).abs() < 1e-12);
        assert!(
            (solution.find("P_busy (repairman busy)").unwrap() - 2.0 / 5.0)
                .abs()
                < 1e-12
        );
    }

    #[test]
    fn repaired_per_unit_time_equals_broken_machines_flow() {
        // every working machine breaks at rate lambda, so
        // A = lambda * (n - w)
        let (n, lambda) = (5.0, 0.2);
        let solution = solve(&[n, lambda, 1.0]).unwrap();

        let a = solution
            .find("A (machines repaired per unit time)")
            .unwrap();
        let w = solution.find("w (mean broken machines)").unwrap();
        assert!((a - lambda * (n - w)).abs() < 1e-9);
    }
}

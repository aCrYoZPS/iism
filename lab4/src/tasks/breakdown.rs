use crate::{
    chain::Chain,
    task::{Param, Solution, Task, require_positive},
};

/// One channel with a queue of m places. A working (busy) channel fails at
/// rate `failure` and is repaired at rate `repair`. The request being served
/// when the channel fails joins the queue if it has a free place and is lost
/// otherwise.
///
/// States: `up k` — channel works with k requests in the system
/// (k = 0..=m+1), `down k` — channel is repaired and k requests wait.
pub fn task() -> Task {
    Task {
        name: "Channel failures",
        params: vec![
            Param::real("λ", 2.0),
            Param::real("μ", 1.0),
            Param::real("failure rate Z", 0.5),
            Param::real("repair rate R", 1.0),
            Param::count("m", 2, 0),
        ],
        solve,
    }
}

fn solve(params: &[f64]) -> Result<Solution, String> {
    let &[lambda, mu, failure, repair, m] = params else {
        return Err("wrong number of parameters".into());
    };
    require_positive("λ", lambda)?;
    require_positive("μ", mu)?;
    require_positive("Z", failure)?;
    require_positive("R", repair)?;
    let m = m as usize;

    // With m = 0 the failed request is lost and the channel is repaired with
    // an empty queue; otherwise `down 0` can never be reached.
    let first_down = m.min(1);
    let up = |k: usize| k;
    let down = |k: usize| m + 2 + k - first_down;
    let states = m + 2 + (m + 1 - first_down);

    let mut chain = Chain::new(states);
    for k in 0..=m + 1 {
        if k <= m {
            chain.add_rate(up(k), up(k + 1), lambda);
        }
        if k >= 1 {
            chain.add_rate(up(k), up(k - 1), mu);
            chain.add_rate(up(k), down(k.min(m)), failure);
        }
    }
    for k in first_down..=m {
        if k < m {
            chain.add_rate(down(k), down(k + 1), lambda);
        }
        chain.add_rate(down(k), up(k), repair);
    }
    let p = chain
        .stationary()
        .ok_or("the chain has no unique solution")?;

    let mut labels: Vec<String> =
        (0..=m + 1).map(|k| format!("up {k}")).collect();
    labels.extend((first_down..=m).map(|k| format!("down {k}")));

    let in_system: f64 = (0..=m + 1)
        .map(|k| k as f64 * p[up(k)])
        .chain((first_down..=m).map(|k| k as f64 * p[down(k)]))
        .sum();
    let serving: f64 = (1..=m + 1).map(|k| p[up(k)]).sum();
    let repairing: f64 = (first_down..=m).map(|k| p[down(k)]).sum();
    let blocked = p[up(m + 1)] + p[down(m)];

    let throughput = mu * serving;
    let accepted = lambda * (1.0 - blocked);

    Ok(Solution::new(labels, p)
        .metric("Q (relative throughput)", throughput / lambda)
        .metric("A (absolute throughput)", throughput)
        .metric("P(channel is being repaired)", repairing)
        .metric("Mean requests in system", in_system)
        .metric(
            "Mean time in system (accepted requests)",
            in_system / accepted,
        ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tasks::queue;

    #[test]
    fn without_failures_it_is_the_plain_queue() {
        // failure rate -> 0 turns the model into M/M/1 with m places
        let broken = solve(&[2.0, 1.0, 1e-12, 1.0, 2.0]).unwrap();
        let plain = (queue::task().solve)(&[2.0, 1.0, 1.0, 2.0, 0.0]).unwrap();

        let a = broken.find("A (absolute throughput)").unwrap();
        let b = plain.find("A (absolute throughput)").unwrap();
        assert!((a - b).abs() < 1e-6, "{a} {b}");
    }

    #[test]
    fn all_cases_of_m_give_a_distribution() {
        for m in 0..5 {
            let solution = solve(&[2.0, 1.0, 0.5, 1.0, m as f64]).unwrap();

            assert!((solution.p.iter().sum::<f64>() - 1.0).abs() < 1e-9);
            assert_eq!(solution.p.len(), solution.labels.len());
        }
    }

    #[test]
    fn repair_time_share_matches_a_failure_repair_cycle() {
        // the channel fails only while it is busy, so the time share in
        // repair is Z * P(busy) / R
        let (failure, repair) = (0.5, 1.0);
        let solution = solve(&[2.0, 1.0, failure, repair, 2.0]).unwrap();

        let repairing = solution.find("P(channel is being repaired)").unwrap();
        let busy: f64 = solution.p[1..=3].iter().sum();
        assert!((repairing - failure * busy / repair).abs() < 1e-9);
    }
}

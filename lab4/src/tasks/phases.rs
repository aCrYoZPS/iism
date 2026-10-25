use crate::{
    chain::Chain,
    task::{Param, Solution, Task, require_positive},
};

/// Customers kept when the queue is unbounded. A state is a (customers,
/// phase) pair, so the chain has up to 1 + TRUNCATION * MAX_PHASES states.
const TRUNCATION: usize = 100;
const TAIL_LIMIT: f64 = 1e-9;
const MAX_PHASES: usize = 5;

/// One channel whose service is a sequence of exponential phases with their
/// own rates (Erlang when the rates are equal, generalised Erlang otherwise).
/// State 0 is an idle channel, state (j, i) is j requests in the system with
/// the served one in phase i.
pub fn task() -> Task {
    Task {
        name: "Phase-type service",
        params: vec![
            Param::real("λ", 0.1),
            Param::count("phases k", 3, 1),
            Param::real("ν1", 0.375),
            Param::real("ν2", 0.375),
            Param::real("ν3", 0.375),
            Param::real("ν4 (if k ≥ 4)", 1.0),
            Param::real("ν5 (if k = 5)", 1.0),
            Param::count("m", 0, 0),
            Param::flag("unbounded queue", true),
        ],
        solve,
    }
}

fn solve(params: &[f64]) -> Result<Solution, String> {
    let &[lambda, phases, nu1, nu2, nu3, nu4, nu5, m, unbounded] = params
    else {
        return Err("wrong number of parameters".into());
    };
    require_positive("λ", lambda)?;
    let (phases, m, unbounded) = (phases as usize, m as usize, unbounded > 0.5);
    if phases > MAX_PHASES {
        return Err(format!("at most {MAX_PHASES} phases are supported"));
    }
    let nu = &[nu1, nu2, nu3, nu4, nu5][..phases];
    for (i, &rate) in nu.iter().enumerate() {
        require_positive(&format!("ν{}", i + 1), rate)?;
    }

    let service_time: f64 = nu.iter().map(|rate| 1.0 / rate).sum();
    if unbounded && lambda * service_time >= 1.0 {
        return Err(format!(
            "λ·t_service ≥ 1: the queue grows without limit, no stationary regime (ρ = {:.4})",
            lambda * service_time
        ));
    }

    let last = if unbounded { TRUNCATION } else { m + 1 };
    let state = |j: usize, i: usize| 1 + (j - 1) * phases + i;
    let mut chain = Chain::new(1 + last * phases);

    chain.add_rate(0, state(1, 0), lambda);
    for j in 1..=last {
        for (i, &rate) in nu.iter().enumerate() {
            let from = state(j, i);
            if j < last {
                chain.add_rate(from, state(j + 1, i), lambda);
            }
            if i + 1 < phases {
                chain.add_rate(from, state(j, i + 1), rate);
            } else if j > 1 {
                chain.add_rate(from, state(j - 1, 0), rate);
            } else {
                chain.add_rate(from, 0, rate);
            }
        }
    }
    let p = chain
        .stationary()
        .ok_or("the chain has no unique solution")?;

    let customers = |j: usize| (j - 1) as f64;
    let mut in_system = 0.0;
    let mut in_queue = 0.0;
    let mut full = 0.0;
    for j in 1..=last {
        for i in 0..phases {
            let pk = p[state(j, i)];
            in_system += j as f64 * pk;
            in_queue += customers(j) * pk;
            if j == last {
                full += pk;
            }
        }
    }

    let reject = if unbounded { 0.0 } else { full };
    let throughput = lambda * (1.0 - reject);

    let mut labels = vec!["0".to_string()];
    for j in 1..=last {
        labels.extend((1..=phases).map(|i| format!("{j},{i}")));
    }
    let idle = p[0];
    let mut solution = Solution::new(labels, p)
        .metric("P_reject", reject)
        .metric("Q (relative throughput)", 1.0 - reject)
        .metric("A (absolute throughput)", throughput)
        .metric("P_busy (channel busy)", 1.0 - idle)
        .metric("Mean requests in system", in_system)
        .metric("Mean queue length", in_queue)
        .metric("Mean time in system", in_system / throughput)
        .metric("Mean time in queue", in_queue / throughput)
        .metric("Mean service time", service_time);

    if unbounded && full > TAIL_LIMIT {
        solution = solution.warning(format!(
            "the queue was cut at {TRUNCATION} requests and their probability is {full:.2e}: results are inaccurate"
        ));
    }
    Ok(solution)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tasks::queue;

    fn run(params: &[f64]) -> Solution {
        solve(params).unwrap()
    }

    #[test]
    fn loss_system_is_insensitive_to_the_service_law() {
        // no queue: P_reject = a / (1 + a) with a = lambda * mean service time
        let (lambda, nu1, nu2) = (0.7, 2.0, 0.5);
        let a = lambda * (1.0 / nu1 + 1.0 / nu2);

        let solution = run(&[lambda, 2.0, nu1, nu2, 1.0, 1.0, 1.0, 0.0, 0.0]);

        let reject = solution.find("P_reject").unwrap();
        assert!((reject - a / (1.0 + a)).abs() < 1e-12, "{reject}");
    }

    #[test]
    fn one_phase_is_the_plain_single_channel_queue() {
        let (lambda, mu) = (0.8, 1.5);

        let phases = run(&[lambda, 1.0, mu, 1.0, 1.0, 1.0, 1.0, 2.0, 0.0]);
        let plain =
            (queue::task().solve)(&[lambda, mu, 1.0, 2.0, 0.0]).unwrap();

        for name in ["A (absolute throughput)", "Mean requests in system"] {
            let (a, b) =
                (phases.find(name).unwrap(), plain.find(name).unwrap());
            assert!((a - b).abs() < 1e-10, "{name}: {a} {b}");
        }
    }

    /// Pollaczek-Khinchine: Lq = (lambda^2 var + rho^2) / (2 (1 - rho)).
    fn pollaczek_khinchine(lambda: f64, rates: &[f64]) -> f64 {
        let mean: f64 = rates.iter().map(|r| 1.0 / r).sum();
        let variance: f64 = rates.iter().map(|r| 1.0 / (r * r)).sum();
        let rho = lambda * mean;
        (lambda * lambda * variance + rho * rho) / (2.0 * (1.0 - rho))
    }

    #[test]
    fn computer_with_erlang_3_service_matches_pollaczek_khinchine() {
        // lambda = 1/10 per minute, Erlang 3 with mean 8 minutes
        let solution =
            run(&[0.1, 3.0, 0.375, 0.375, 0.375, 1.0, 1.0, 0.0, 1.0]);

        let expected = pollaczek_khinchine(0.1, &[0.375; 3]);
        let queue = solution.find("Mean queue length").unwrap();
        assert!((expected - 2.1333333).abs() < 1e-6);
        assert!((queue - expected).abs() < 1e-4, "{queue} {expected}");
        assert!(solution.warnings.is_empty());
    }

    #[test]
    fn unequal_phase_rates_match_pollaczek_khinchine() {
        let solution = run(&[0.3, 3.0, 1.0, 2.0, 4.0, 1.0, 1.0, 0.0, 1.0]);

        let expected = pollaczek_khinchine(0.3, &[1.0, 2.0, 4.0]);
        let queue = solution.find("Mean queue length").unwrap();
        assert!((queue - expected).abs() < 1e-6, "{queue} {expected}");
    }

    #[test]
    fn overloaded_unbounded_queue_is_rejected() {
        assert!(
            solve(&[0.2, 3.0, 0.375, 0.375, 0.375, 1.0, 1.0, 0.0, 1.0])
                .is_err()
        );
    }

    #[test]
    fn states_are_a_distribution_for_every_phase_count() {
        for phases in 1..=5 {
            let solution =
                run(&[0.2, phases as f64, 2.0, 3.0, 4.0, 5.0, 6.0, 2.0, 0.0]);

            assert!((solution.p.iter().sum::<f64>() - 1.0).abs() < 1e-9);
            assert_eq!(solution.p.len(), solution.labels.len());
            assert_eq!(solution.p.len(), 1 + 3 * phases);
        }
    }
}

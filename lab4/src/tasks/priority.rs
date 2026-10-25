use crate::{
    chain::Chain,
    task::{Param, Solution, Task, require_positive},
};

const CHANNELS: usize = 2;

/// Two-channel system with refusals and two request flows. A type I request
/// pushes a type II one out of service when all channels are busy.
/// State (r, j): r requests of type I and j of type II in service.
pub fn task() -> Task {
    Task {
        name: "Priority with preemption",
        params: vec![
            Param::real("λ1", 1.0),
            Param::real("λ2", 2.0),
            Param::real("μ1", 1.0),
            Param::real("μ2", 1.0),
        ],
        solve,
    }
}

fn solve(params: &[f64]) -> Result<Solution, String> {
    let &[lambda1, lambda2, mu1, mu2] = params else {
        return Err("wrong number of parameters".into());
    };
    require_positive("λ1", lambda1)?;
    require_positive("λ2", lambda2)?;
    require_positive("μ1", mu1)?;
    require_positive("μ2", mu2)?;

    let states: Vec<(usize, usize)> = (0..=CHANNELS)
        .flat_map(|r| (0..=CHANNELS - r).map(move |j| (r, j)))
        .collect();
    let index = |r: usize, j: usize| {
        states.iter().position(|&state| state == (r, j)).unwrap()
    };

    let mut chain = Chain::new(states.len());
    for (from, &(r, j)) in states.iter().enumerate() {
        if r + j < CHANNELS {
            chain.add_rate(from, index(r + 1, j), lambda1);
            chain.add_rate(from, index(r, j + 1), lambda2);
        } else if j > 0 {
            // all channels busy, type II is pushed out
            chain.add_rate(from, index(r + 1, j - 1), lambda1);
        }
        if r > 0 {
            chain.add_rate(from, index(r - 1, j), r as f64 * mu1);
        }
        if j > 0 {
            chain.add_rate(from, index(r, j - 1), j as f64 * mu2);
        }
    }
    let p = chain
        .stationary()
        .ok_or("the chain has no unique solution")?;

    let sum = |keep: &dyn Fn(usize, usize) -> f64| -> f64 {
        states
            .iter()
            .zip(&p)
            .map(|(&(r, j), pk)| keep(r, j) * pk)
            .sum()
    };
    let reject1 = sum(&|r, _| (r == CHANNELS) as u8 as f64);
    let reject2_on_arrival = sum(&|r, j| (r + j == CHANNELS) as u8 as f64);
    let served1 = lambda1 * (1.0 - reject1);
    let served2 = mu2 * sum(&|_, j| j as f64);
    let busy = sum(&|r, j| (r + j) as f64);

    let labels = states.iter().map(|(r, j)| format!("({r},{j})")).collect();
    Ok(Solution::new(labels, p)
        .metric("P_reject, type I", reject1)
        .metric("P_reject, type II on arrival", reject2_on_arrival)
        .metric(
            "Type II not served (refused or pushed out)",
            1.0 - served2 / lambda2,
        )
        .metric("A1 (type I served per unit time)", served1)
        .metric("A2 (type II served per unit time)", served2)
        .metric("Mean busy channels", busy))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn type_i_alone_sees_an_erlang_loss_system() {
        // lambda2 is almost zero, so type I does not notice type II:
        // rho = 1.5, B = (rho^2/2) / (1 + rho + rho^2/2)
        let rho = 1.5_f64;
        let expected = (rho * rho / 2.0) / (1.0 + rho + rho * rho / 2.0);

        let solution = solve(&[1.5, 1e-12, 1.0, 1.0]).unwrap();

        let reject = solution.find("P_reject, type I").unwrap();
        assert!((reject - expected).abs() < 1e-6, "{reject} {expected}");
    }

    #[test]
    fn type_i_served_flows_agree() {
        // type I is never pushed out, so what arrives and is accepted is
        // exactly what completes: lambda1 (1 - P) = mu1 * E[r]
        let (lambda1, mu1) = (1.0, 1.5);
        let solution = solve(&[lambda1, 2.0, mu1, 1.0]).unwrap();

        let served = solution.find("A1 (type I served per unit time)").unwrap();
        let in_service: f64 = solution
            .p
            .iter()
            .zip([(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (2, 0)])
            .map(|(pk, (r, _))| r as f64 * pk)
            .sum();
        assert!((served - mu1 * in_service).abs() < 1e-9);
    }

    #[test]
    fn busy_channels_balance_with_served_flows() {
        // each type's served rate times its mean service time is its load
        let (mu1, mu2) = (1.0, 0.5);
        let solution = solve(&[1.0, 2.0, mu1, mu2]).unwrap();

        let a1 = solution.find("A1 (type I served per unit time)").unwrap();
        let a2 = solution.find("A2 (type II served per unit time)").unwrap();
        let busy = solution.find("Mean busy channels").unwrap();
        assert!((busy - (a1 / mu1 + a2 / mu2)).abs() < 1e-9);
    }
}

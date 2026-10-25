use nalgebra::{DMatrix, DVector};

/// Continuous-time Markov chain given by the rates of transitions between
/// its states.
pub struct Chain {
    rates: DMatrix<f64>,
}

impl Chain {
    pub fn new(states: usize) -> Self {
        Self {
            rates: DMatrix::zeros(states, states),
        }
    }

    pub fn add_rate(&mut self, from: usize, to: usize, rate: f64) {
        debug_assert_ne!(from, to);
        self.rates[(from, to)] += rate;
    }

    /// Final probabilities p: pQ = 0 and sum(p) = 1, where Q is the rate
    /// matrix with minus the row sum on the diagonal. `None` when the
    /// equations have no unique solution.
    pub fn stationary(&self) -> Option<Vec<f64>> {
        let n = self.rates.nrows();

        let mut q = self.rates.clone();
        for i in 0..n {
            q[(i, i)] = -self.rates.row(i).sum();
        }

        // pQ = 0 is Q^T p^T = 0; one balance equation follows from the
        // others, so it is replaced by the normalisation sum(p) = 1.
        let mut a = q.transpose();
        a.row_mut(n - 1).fill(1.0);

        let mut b = DVector::zeros(n);
        b[n - 1] = 1.0;

        let p = a.lu().solve(&b)?;
        p.iter()
            .all(|x| x.is_finite())
            .then(|| p.as_slice().to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_state_chain_matches_the_hand_solution() {
        // task 1: lambda = 0.4, mu = 1/3  =>  p = (5/11, 6/11)
        let mut chain = Chain::new(2);
        chain.add_rate(0, 1, 0.4);
        chain.add_rate(1, 0, 1.0 / 3.0);

        let p = chain.stationary().unwrap();

        assert!((p[0] - 5.0 / 11.0).abs() < 1e-12, "{p:?}");
        assert!((p[1] - 6.0 / 11.0).abs() < 1e-12, "{p:?}");
    }

    #[test]
    fn disconnected_chain_has_no_unique_solution() {
        let mut chain = Chain::new(3);
        chain.add_rate(0, 1, 1.0);
        chain.add_rate(1, 0, 1.0);

        assert_eq!(chain.stationary(), None);
    }
}

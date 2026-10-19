/// The discrete distribution under study: outcomes and their probabilities.
/// To switch distributions, rewrite this file.
pub struct DiscreteParams {}

impl DiscreteParams {
    pub const VALUES: [f64; 10] =
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

    pub const PROBABILITIES: [f64; 10] =
        [0.05, 0.12, 0.08, 0.15, 0.10, 0.20, 0.06, 0.09, 0.07, 0.08];

    pub fn mean() -> f64 {
        Self::VALUES
            .iter()
            .zip(Self::PROBABILITIES)
            .map(|(y, p)| y * p)
            .sum()
    }

    pub fn variance() -> f64 {
        let mean = Self::mean();

        Self::VALUES
            .iter()
            .zip(Self::PROBABILITIES)
            .map(|(y, p)| (y - mean).powi(2) * p)
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probabilities_sum_to_one() {
        let sum: f64 = DiscreteParams::PROBABILITIES.iter().sum();

        assert!((sum - 1.0).abs() < 1e-12);
    }

    #[test]
    fn mean_and_variance_follow_the_definition() {
        // sum(y * p) = 5.4, sum(y^2 * p) = 35.72
        assert!((DiscreteParams::mean() - 5.4).abs() < 1e-9);
        assert!(
            (DiscreteParams::variance() - (35.72 - 5.4 * 5.4)).abs() < 1e-9
        );
    }
}

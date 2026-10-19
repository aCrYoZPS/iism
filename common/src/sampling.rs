pub fn is_hit(draw: f64, probability: f64) -> bool {
    draw < probability
}

pub fn cumulative(probabilities: &[f64]) -> Vec<f64> {
    let mut sum = 0.0;
    let mut cumulative: Vec<f64> = probabilities
        .iter()
        .map(|probability| {
            sum += probability;
            sum
        })
        .collect();

    if let Some(last) = cumulative.last_mut() {
        *last = 1.0;
    }

    cumulative
}

pub fn pick_index(cumulative: &[f64], draw: f64) -> usize {
    cumulative
        .iter()
        .position(|upper_bound| draw < *upper_bound)
        .unwrap_or(cumulative.len().saturating_sub(1))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pick_index_follows_cumulative_bounds() {
        let cumulative = cumulative(&[0.2, 0.5, 0.3]);

        assert_eq!(pick_index(&cumulative, 0.0), 0);
        assert_eq!(pick_index(&cumulative, 0.19), 0);
        assert_eq!(pick_index(&cumulative, 0.2), 1);
        assert_eq!(pick_index(&cumulative, 0.69), 1);
        assert_eq!(pick_index(&cumulative, 0.71), 2);
        assert_eq!(pick_index(&cumulative, 0.999_999), 2);
    }

    #[test]
    fn frequencies_match_probabilities_over_10000_runs() {
        let probabilities = [0.1, 0.2, 0.3, 0.4];
        let cumulative = cumulative(&probabilities);
        let mut counts = [0u64; 4];

        for _ in 0..10_000 {
            counts[pick_index(&cumulative, rand::random())] += 1;
        }

        let test = crate::stats::chi_square(&counts, &probabilities)
            .expect("enough categories");

        assert!(test.statistic < 3.0 * test.critical, "{test:?}");
    }
}

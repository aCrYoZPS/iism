const Z_95: f64 = 1.96;

#[derive(Debug, Clone, Copy)]
pub struct Estimates {
    pub mean: f64,
    pub variance: f64,
    pub mean_interval: (f64, f64),
    pub variance_interval: (f64, f64),
}

pub fn estimates(sample: &[f64]) -> Option<Estimates> {
    if sample.len() < 2 {
        return None;
    }

    let n = sample.len() as f64;
    let mean = sample.iter().sum::<f64>() / n;
    let variance =
        sample.iter().map(|y| (y - mean).powi(2)).sum::<f64>() / (n - 1.0);

    let half_width = Z_95 * (variance / n).sqrt();

    let dof = n - 1.0;
    let variance_interval = (
        dof * variance / chi_square_quantile(dof, Z_95),
        dof * variance / chi_square_quantile(dof, -Z_95),
    );

    Some(Estimates {
        mean,
        variance,
        mean_interval: (mean - half_width, mean + half_width),
        variance_interval,
    })
}

fn chi_square_quantile(dof: f64, z: f64) -> f64 {
    let a = 2.0 / (9.0 * dof);
    dof * (1.0 - a + z * a.sqrt()).powi(3)
}

pub fn histogram(sample: &[f64], low: f64, high: f64, bins: usize) -> Vec<u64> {
    let width = (high - low) / bins as f64;
    let mut counts = vec![0u64; bins];

    for &y in sample {
        if (low..high).contains(&y) {
            let index = (((y - low) / width) as usize).min(bins - 1);
            counts[index] += 1;
        }
    }

    counts
}

pub fn class_counts(sample: &[f64], edges: &[f64]) -> Vec<u64> {
    let mut counts = vec![0u64; edges.len() + 1];

    for &y in sample {
        counts[edges.partition_point(|&edge| edge <= y)] += 1;
    }

    counts
}

pub fn empirical_cdf(sample: &[f64]) -> Vec<[f64; 2]> {
    let mut sorted = sample.to_vec();
    sorted.sort_by(f64::total_cmp);

    let n = sorted.len() as f64;
    sorted
        .into_iter()
        .enumerate()
        .map(|(i, y)| [y, (i + 1) as f64 / n])
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn estimates_of_a_small_sample() {
        let result = estimates(&[1.0, 2.0, 3.0, 4.0]).unwrap();

        assert!((result.mean - 2.5).abs() < 1e-12);
        assert!((result.variance - 5.0 / 3.0).abs() < 1e-12);
        assert!(result.mean_interval.0 < result.mean);
        assert!(result.mean < result.mean_interval.1);
        assert!(result.variance_interval.0 < result.variance);
        assert!(result.variance < result.variance_interval.1);
        assert!(estimates(&[1.0]).is_none());
    }

    #[test]
    fn chi_square_quantile_is_close_to_the_table() {
        // Table values for 100 degrees of freedom: 74.22 and 129.56.
        assert!((chi_square_quantile(100.0, -Z_95) - 74.22).abs() < 0.5);
        assert!((chi_square_quantile(100.0, Z_95) - 129.56).abs() < 0.5);
    }

    #[test]
    fn histogram_ignores_values_outside_the_range() {
        let counts =
            histogram(&[-1.0, 0.0, 0.4, 0.5, 0.99, 1.0, 7.0], 0.0, 1.0, 2);

        assert_eq!(counts, vec![2, 2]);
    }

    #[test]
    fn class_counts_cover_the_whole_axis() {
        let counts =
            class_counts(&[-5.0, 0.5, 1.0, 1.5, 2.0, 9.0], &[1.0, 2.0]);

        assert_eq!(counts, vec![2, 2, 2]);
    }

    #[test]
    fn empirical_cdf_is_sorted_and_ends_at_one() {
        let points = empirical_cdf(&[3.0, 1.0, 2.0]);

        assert_eq!(
            points,
            vec![[1.0, 1.0 / 3.0], [2.0, 2.0 / 3.0], [3.0, 1.0]]
        );
    }
}

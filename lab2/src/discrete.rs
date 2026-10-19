use common::{
    colors, sampling,
    stats::{self, ChiSquare},
    tab::Tab,
};
use eframe::egui;
use egui_plot::{Line, PlotPoints};

use crate::{
    analysis::{self, Estimates},
    discrete_params::DiscreteParams,
    report,
};

const MIN_EXPECTED: f64 = 5.0;

struct Report {
    runs: u32,
    counts: Vec<u64>,
    estimates: Option<Estimates>,
    chi_square: Option<ChiSquare>,
}

pub struct DiscreteTab {
    runs: u32,
    report: Option<Report>,
}

impl DiscreteTab {
    pub fn new() -> Self {
        Self {
            runs: stats::DEFAULT_RUNS,
            report: None,
        }
    }

    fn run(&mut self) {
        let cumulative = sampling::cumulative(&DiscreteParams::PROBABILITIES);
        let mut counts = vec![0u64; DiscreteParams::VALUES.len()];
        let mut sample = Vec::with_capacity(self.runs as usize);

        for _ in 0..self.runs {
            let index = sampling::pick_index(&cumulative, rand::random());
            counts[index] += 1;
            sample.push(DiscreteParams::VALUES[index]);
        }

        self.report = Some(Report {
            runs: self.runs,
            chi_square: stats::chi_square(
                &counts,
                &DiscreteParams::PROBABILITIES,
            ),
            estimates: analysis::estimates(&sample),
            counts,
        });
    }
}

/// Points of a step function that jumps to `levels[i]` at `values[i]`.
fn step_points(values: &[f64], levels: &[f64]) -> Vec<[f64; 2]> {
    let mut points = vec![[values[0] - 1.0, 0.0]];
    let mut previous = 0.0;

    for (&value, &level) in values.iter().zip(levels) {
        points.push([value, previous]);
        points.push([value, level]);
        previous = level;
    }

    points.push([values[values.len() - 1] + 1.0, previous]);
    points
}

fn show_cdf(ui: &mut egui::Ui, id: &str, counts: &[u64], runs: u32) {
    let mut sum = 0u64;
    let empirical: Vec<f64> = counts
        .iter()
        .map(|&count| {
            sum += count;
            sum as f64 / runs as f64
        })
        .collect();
    let theoretical = sampling::cumulative(&DiscreteParams::PROBABILITIES);

    report::plot(id).show(ui, |plot_ui| {
        plot_ui.line(
            Line::new(
                "empirical",
                PlotPoints::from(step_points(
                    &DiscreteParams::VALUES,
                    &empirical,
                )),
            )
            .color(colors::BLUE),
        );
        plot_ui.line(
            Line::new(
                "G(y)",
                PlotPoints::from(step_points(
                    &DiscreteParams::VALUES,
                    &theoretical,
                )),
            )
            .color(colors::ORANGE),
        );
    });
}

fn show_report(ui: &mut egui::Ui, report: &Report) {
    ui.heading(format!("Statistics ({} runs)", report.runs));

    let labels: Vec<String> = DiscreteParams::VALUES
        .iter()
        .map(|y| y.to_string())
        .collect();

    stats::show_frequency_report(
        ui,
        "discrete_frequencies",
        report.runs as u64,
        &labels,
        &DiscreteParams::PROBABILITIES,
        &report.counts,
    );

    ui.add_space(8.0);

    ui.label("Empirical and theoretical distribution function");
    show_cdf(ui, "discrete_cdf", &report.counts, report.runs);

    ui.add_space(8.0);

    if let Some(estimates) = &report.estimates {
        report::show_estimates(
            ui,
            estimates,
            DiscreteParams::mean(),
            DiscreteParams::variance(),
        );
    }

    ui.add_space(8.0);

    if let Some(test) = &report.chi_square {
        stats::show_chi_square(ui, test);
    }

    let smallest = DiscreteParams::PROBABILITIES
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    let expected = report.runs as f64 * smallest;
    if expected < MIN_EXPECTED {
        ui.label(
            egui::RichText::new(format!(
                "Only {expected:.1} values expected for the rarest outcome: the \u{03c7}\u{00b2} test is unreliable"
            ))
            .color(colors::RED),
        );
    }
}

impl Tab for DiscreteTab {
    fn name(&self) -> &'static str {
        "Discrete"
    }

    fn ui(&mut self, ui: &mut egui::Ui) {
        ui.heading(self.name());

        ui.add_space(8.0);

        if stats::runs_control(ui, &mut self.runs) {
            self.run();
        }

        ui.add_space(8.0);

        if let Some(report) = &self.report {
            egui::ScrollArea::vertical().show(ui, |ui| {
                show_report(ui, report);
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn step_points_jump_at_each_value() {
        let points = step_points(&[1.0, 2.0], &[0.25, 1.0]);

        assert_eq!(
            points,
            vec![
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 0.25],
                [2.0, 0.25],
                [2.0, 1.0],
                [3.0, 1.0],
            ]
        );
    }

    #[test]
    fn generated_sample_matches_the_distribution() {
        let cumulative = sampling::cumulative(&DiscreteParams::PROBABILITIES);
        let mut counts = vec![0u64; DiscreteParams::VALUES.len()];

        for _ in 0..10_000 {
            counts[sampling::pick_index(&cumulative, rand::random())] += 1;
        }

        let test =
            stats::chi_square(&counts, &DiscreteParams::PROBABILITIES).unwrap();
        assert!(test.statistic < 3.0 * test.critical, "{test:?}");
    }
}

use common::{
    colors,
    stats::{self, ChiSquare},
    tab::Tab,
};
use eframe::egui;
use egui_plot::{Bar, BarChart, Line, PlotPoints};

use crate::{
    analysis::{self, Estimates},
    params::Params,
    report,
};

const HISTOGRAM_BINS: usize = 50;
const CLASSES: usize = 50;
const MIN_EXPECTED: f64 = 5.0;
const CURVE_POINTS: usize = 200;

struct Report {
    runs: u32,
    heights: Vec<f64>,
    ecdf: Vec<[f64; 2]>,
    estimates: Option<Estimates>,
    chi_square: Option<ChiSquare>,
}

pub struct ContinuousTab {
    runs: u32,
    report: Option<Report>,
}

impl ContinuousTab {
    pub fn new() -> Self {
        Self {
            runs: stats::DEFAULT_RUNS,
            report: None,
        }
    }

    fn run(&mut self) {
        let sample: Vec<f64> = (0..self.runs)
            .map(|_| Params::g_inv(rand::random()))
            .collect();

        let (low, high) = Params::range();
        let width = (high - low) / HISTOGRAM_BINS as f64;
        let heights = analysis::histogram(&sample, low, high, HISTOGRAM_BINS)
            .into_iter()
            .map(|count| count as f64 / (self.runs as f64 * width))
            .collect();

        let edges: Vec<f64> = (1..CLASSES)
            .map(|j| Params::g_inv(j as f64 / CLASSES as f64))
            .collect();
        let counts = analysis::class_counts(&sample, &edges);
        let probabilities = vec![1.0 / CLASSES as f64; CLASSES];

        self.report = Some(Report {
            runs: self.runs,
            heights,
            ecdf: analysis::empirical_cdf(&sample),
            estimates: analysis::estimates(&sample),
            chi_square: stats::chi_square(&counts, &probabilities),
        });
    }
}

fn curve(f: fn(f64) -> f64) -> Vec<[f64; 2]> {
    let (low, high) = Params::range();

    (0..=CURVE_POINTS)
        .map(|i| {
            let x = low + (high - low) * i as f64 / CURVE_POINTS as f64;
            [x, f(x)]
        })
        .collect()
}

fn show_density(ui: &mut egui::Ui, id: &str, heights: &[f64]) {
    let (low, high) = Params::range();
    let width = (high - low) / HISTOGRAM_BINS as f64;

    let bars = heights
        .iter()
        .enumerate()
        .map(|(i, &height)| {
            Bar::new(low + (i as f64 + 0.5) * width, height).width(width)
        })
        .collect();

    report::plot(id).show(ui, |plot_ui| {
        plot_ui.bar_chart(BarChart::new("histogram", bars).color(colors::BLUE));
        plot_ui.line(
            Line::new("f(y)", PlotPoints::from(curve(Params::f)))
                .color(colors::ORANGE),
        );
    });
}

fn show_cdf(ui: &mut egui::Ui, id: &str, ecdf: &[[f64; 2]]) {
    let (low, high) = Params::range();
    let visible: Vec<[f64; 2]> = ecdf
        .iter()
        .copied()
        .filter(|[y, _]| (low..=high).contains(y))
        .collect();

    report::plot(id).show(ui, |plot_ui| {
        plot_ui.line(
            Line::new("empirical", PlotPoints::from(visible))
                .color(colors::BLUE),
        );
        plot_ui.line(
            Line::new("G(y)", PlotPoints::from(curve(Params::g)))
                .color(colors::ORANGE),
        );
    });
}

fn show_report(ui: &mut egui::Ui, report: &Report) {
    ui.heading(format!("Statistics ({} runs)", report.runs));

    ui.label("Histogram and density");
    show_density(ui, "continuous_density", &report.heights);

    ui.add_space(8.0);

    ui.label("Empirical and theoretical distribution function");
    show_cdf(ui, "continuous_cdf", &report.ecdf);

    ui.add_space(8.0);

    if let Some(estimates) = &report.estimates {
        report::show_estimates(
            ui,
            estimates,
            Params::mean(),
            Params::variance(),
        );
    }

    ui.add_space(8.0);

    if let Some(test) = &report.chi_square {
        stats::show_chi_square(ui, test);
    }

    let expected = report.runs as f64 / CLASSES as f64;
    if expected < MIN_EXPECTED {
        ui.label(
            egui::RichText::new(format!(
                "Only {expected:.1} values expected per class: the \u{03c7}\u{00b2} test is unreliable"
            ))
            .color(colors::RED),
        );
    }
}

impl Tab for ContinuousTab {
    fn name(&self) -> &'static str {
        "Continuous"
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
    fn generated_sample_matches_the_distribution() {
        let sample: Vec<f64> =
            (0..10_000).map(|_| Params::g_inv(rand::random())).collect();

        let edges: Vec<f64> = (1..CLASSES)
            .map(|j| Params::g_inv(j as f64 / CLASSES as f64))
            .collect();
        let counts = analysis::class_counts(&sample, &edges);
        let probabilities = vec![1.0 / CLASSES as f64; CLASSES];
        let test = stats::chi_square(&counts, &probabilities).unwrap();
        assert!(test.statistic < 3.0 * test.critical, "{test:?}");

        let estimates = analysis::estimates(&sample).unwrap();
        assert!(
            (estimates.mean - Params::mean()).abs() < 0.05,
            "{estimates:?}"
        );
        assert!(
            (estimates.variance - Params::variance()).abs() < 0.05,
            "{estimates:?}"
        );
    }
}

use std::ops::RangeInclusive;

use eframe::egui::{self, Color32};
use egui_plot::{
    Bar, BarChart, GridMark, HLine, Legend, Line, Plot, PlotPoints,
};

use crate::colors;

pub const DEFAULT_RUNS: u32 = 10_000;
const MAX_RUNS: u32 = 10_000_000;
const CHART_HEIGHT: f32 = 240.0;

pub fn runs_control(ui: &mut egui::Ui, runs: &mut u32) -> bool {
    let mut clicked = false;

    ui.horizontal(|ui| {
        ui.label("Runs:");
        ui.add(egui::DragValue::new(runs).range(1..=MAX_RUNS).speed(100.0));
        clicked = ui.button(format!("Run {} times", *runs)).clicked();
    });

    clicked
}

#[derive(Debug, Clone, Copy)]
pub struct ChiSquare {
    pub statistic: f64,
    pub dof: usize,
    pub critical: f64,
}

impl ChiSquare {
    pub fn is_consistent(&self) -> bool {
        self.statistic <= self.critical
    }
}

pub fn chi_square(counts: &[u64], probabilities: &[f64]) -> Option<ChiSquare> {
    let runs: u64 = counts.iter().sum();
    let mut statistic = 0.0;
    let mut categories = 0usize;

    for (&count, &probability) in counts.iter().zip(probabilities) {
        if probability <= 0.0 {
            continue;
        }

        categories += 1;
        let expected = runs as f64 * probability;
        statistic += (count as f64 - expected).powi(2) / expected;
    }

    if runs == 0 || categories < 2 {
        return None;
    }

    let dof = categories - 1;
    let k = dof as f64;
    let a = 2.0 / (9.0 * k);
    let critical = k * (1.0 - a + 1.645 * a.sqrt()).powi(3);

    Some(ChiSquare {
        statistic,
        dof,
        critical,
    })
}

pub fn show_chi_square(ui: &mut egui::Ui, test: &ChiSquare) {
    let (verdict, color) = if test.is_consistent() {
        ("consistent with the theory at the 5% level", colors::GREEN)
    } else {
        ("differs from the theory at the 5% level", colors::RED)
    };

    ui.label(format!(
        "\u{03c7}\u{00b2} = {:.3} (df = {}, critical \u{2248} {:.3})",
        test.statistic, test.dof, test.critical
    ));
    ui.label(egui::RichText::new(verdict).color(color));
}

pub struct Series<'a> {
    pub name: &'a str,
    pub values: &'a [f64],
    pub color: Color32,
}

pub fn grouped_bar_chart(
    ui: &mut egui::Ui,
    id: &str,
    labels: &[String],
    series: &[Series],
) {
    let width = 0.8 / series.len().max(1) as f64;
    let labels = labels.to_vec();

    Plot::new(id)
        .height(CHART_HEIGHT)
        .legend(Legend::default())
        .include_y(0.0)
        .allow_zoom(false)
        .allow_drag(false)
        .allow_scroll(false)
        .allow_boxed_zoom(false)
        .x_axis_formatter(move |mark: GridMark, _: &RangeInclusive<f64>| {
            let x = mark.value;
            let index = x.round();

            if (x - index).abs() < 1e-6
                && index >= 0.0
                && (index as usize) < labels.len()
            {
                labels[index as usize].clone()
            } else {
                String::new()
            }
        })
        .show(ui, |plot_ui| {
            for (s, item) in series.iter().enumerate() {
                let offset =
                    (s as f64 - (series.len() as f64 - 1.0) / 2.0) * width;

                let bars = item
                    .values
                    .iter()
                    .enumerate()
                    .map(|(i, &value)| {
                        Bar::new(i as f64 + offset, value).width(width)
                    })
                    .collect();

                plot_ui.bar_chart(
                    BarChart::new(item.name, bars).color(item.color),
                );
            }
        });
}

pub fn convergence_chart(
    ui: &mut egui::Ui,
    id: &str,
    points: &[[f64; 2]],
    expected: f64,
) {
    Plot::new(id)
        .height(CHART_HEIGHT)
        .legend(Legend::default())
        .allow_zoom(false)
        .allow_drag(false)
        .allow_scroll(false)
        .allow_boxed_zoom(false)
        .show(ui, |plot_ui| {
            plot_ui.line(
                Line::new("observed", PlotPoints::from(points.to_vec()))
                    .color(colors::BLUE),
            );
            plot_ui
                .hline(HLine::new("expected", expected).color(colors::ORANGE));
        });
}

pub fn show_frequency_report(
    ui: &mut egui::Ui,
    id: &str,
    runs: u64,
    labels: &[String],
    expected: &[f64],
    counts: &[u64],
) {
    let observed: Vec<f64> = counts
        .iter()
        .map(|&count| count as f64 / runs as f64)
        .collect();

    grouped_bar_chart(
        ui,
        id,
        labels,
        &[
            Series {
                name: "expected",
                values: expected,
                color: colors::ORANGE,
            },
            Series {
                name: "observed",
                values: &observed,
                color: colors::BLUE,
            },
        ],
    );

    ui.add_space(8.0);

    egui::Grid::new(format!("{id}_table"))
        .striped(true)
        .spacing([16.0, 6.0])
        .show(ui, |ui| {
            ui.label("Outcome");
            ui.label("Expected");
            ui.label("Observed");
            ui.label("Count");
            ui.label("Abs. error");
            ui.end_row();

            for (i, label) in labels.iter().enumerate() {
                ui.label(label);
                ui.label(format!("{:.4}", expected[i]));
                ui.label(format!("{:.4}", observed[i]));
                ui.label(counts[i].to_string());
                ui.label(format!("{:.4}", (observed[i] - expected[i]).abs()));
                ui.end_row();
            }
        });
}

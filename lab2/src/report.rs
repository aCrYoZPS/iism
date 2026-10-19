use common::colors;
use eframe::egui;
use egui_plot::{Legend, Plot};

use crate::analysis::Estimates;

const CHART_HEIGHT: f32 = 280.0;

pub fn plot(id: &str) -> Plot<'_> {
    Plot::new(id)
        .height(CHART_HEIGHT)
        .legend(Legend::default())
        .include_y(0.0)
        .allow_zoom(false)
        .allow_drag(false)
        .allow_scroll(false)
        .allow_boxed_zoom(false)
}

fn show_interval(
    ui: &mut egui::Ui,
    name: &str,
    estimate: f64,
    interval: (f64, f64),
    theory: f64,
) {
    let (verdict, color) = if (interval.0..=interval.1).contains(&theory) {
        ("theory is inside the interval", colors::GREEN)
    } else {
        ("theory is outside the interval", colors::RED)
    };

    ui.label(format!(
        "{name}: {estimate:.4} · 95% interval [{:.4}; {:.4}] · theory {theory:.4}",
        interval.0, interval.1
    ));
    ui.label(egui::RichText::new(verdict).color(color));
}

pub fn show_estimates(
    ui: &mut egui::Ui,
    estimates: &Estimates,
    theory_mean: f64,
    theory_variance: f64,
) {
    show_interval(
        ui,
        "Mean",
        estimates.mean,
        estimates.mean_interval,
        theory_mean,
    );
    show_interval(
        ui,
        "Variance",
        estimates.variance,
        estimates.variance_interval,
        theory_variance,
    );
}

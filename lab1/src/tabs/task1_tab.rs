use common::{colors, sampling, stats, tab::Tab};
use eframe::egui::{self, text::LayoutJob, text::TextFormat};

const DEFAULT_PROBABILITY: f64 = 0.5;

const MAX_CURVE_POINTS: u32 = 500;

#[derive(Debug)]
struct RunStats {
    runs: u32,
    hits: u32,
    probability: f64,
    curve: Vec<[f64; 2]>,
}

#[derive(Debug)]
pub struct Task1Tab {
    probability: f64,
    last_draw: Option<f64>,
    runs: u32,
    stats: Option<RunStats>,
}

impl Task1Tab {
    pub fn new() -> Self {
        Self {
            probability: DEFAULT_PROBABILITY,
            last_draw: None,
            runs: stats::DEFAULT_RUNS,
            stats: None,
        }
    }

    fn run_many(&mut self) {
        let step = (self.runs / MAX_CURVE_POINTS).max(1);
        let mut hits = 0;
        let mut curve = Vec::new();

        for run in 1..=self.runs {
            if sampling::is_hit(rand::random(), self.probability) {
                hits += 1;
            }

            if run % step == 0 || run == self.runs {
                curve.push([run as f64, hits as f64 / run as f64]);
            }
        }

        self.stats = Some(RunStats {
            runs: self.runs,
            hits,
            probability: self.probability,
            curve,
        });
    }

    fn draw_stats(&self, ui: &mut egui::Ui) {
        let Some(stats) = &self.stats else {
            return;
        };

        let frequency = stats.hits as f64 / stats.runs as f64;
        let standard_error = (stats.probability * (1.0 - stats.probability)
            / stats.runs as f64)
            .sqrt();

        ui.heading(format!("Statistics ({} runs)", stats.runs));
        ui.label(format!(
            "Hits: {} · Observed frequency {:.4} · Probability {:.4}",
            stats.hits, frequency, stats.probability
        ));
        ui.label(format!(
            "Absolute error {:.4} · Standard error {:.4}",
            (frequency - stats.probability).abs(),
            standard_error
        ));

        ui.add_space(8.0);

        stats::convergence_chart(
            ui,
            "task1_convergence",
            &stats.curve,
            stats.probability,
        );
    }
}

impl Tab for Task1Tab {
    fn name(&self) -> &'static str {
        "Task1"
    }

    fn ui(&mut self, ui: &mut egui::Ui) {
        ui.heading(self.name());

        ui.add_space(8.0);

        ui.group(|ui| {
            ui.set_min_width(ui.available_width());
            ui.add(
                egui::Slider::new(&mut self.probability, 0.0..=1.0)
                    .text("Probability (0–1)"),
            );

            ui.add_space(8.0);

            if ui.button("Compute").clicked() {
                self.last_draw = Some(rand::random());
            }

            ui.separator();

            if let Some(draw) = self.last_draw {
                let hit = sampling::is_hit(draw, self.probability);
                let (label, color) = if hit {
                    ("True", colors::GREEN)
                } else {
                    ("False", colors::RED)
                };

                ui.label(
                    egui::RichText::new(format!(
                        "Draw: {draw:.3} · Probability {:.3}",
                        self.probability
                    ))
                    .weak(),
                );

                let font = egui::TextStyle::Body.resolve(ui.style());
                let mut job = LayoutJob::default();
                job.append(
                    "Result: ",
                    0.0,
                    TextFormat::simple(font.clone(), ui.visuals().text_color()),
                );
                job.append(label, 0.0, TextFormat::simple(font, color));

                ui.label(job);
            } else {
                ui.label(egui::RichText::new("-").weak());
                ui.label(egui::RichText::new(" ").weak());
            }

            ui.separator();

            if stats::runs_control(ui, &mut self.runs) {
                self.run_many();
            }

            ui.add_space(8.0);

            self.draw_stats(ui);
        });
    }
}

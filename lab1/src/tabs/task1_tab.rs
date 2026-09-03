use crate::common::{colors, tab::Tab};
use eframe::egui::{self, text::LayoutJob, text::TextFormat};

const DEFAULT_PROBABILITY: f64 = 0.5;

#[derive(Debug)]
pub struct Task1Tab {
    probability: f64,
    last_draw: Option<f64>,
}

impl Task1Tab {
    pub fn new() -> Self {
        Self {
            probability: DEFAULT_PROBABILITY,
            last_draw: None,
        }
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
                let hit = draw < self.probability;
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
        });
    }
}

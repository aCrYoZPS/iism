use std::vec;

use crate::common::{colors, tab::Tab, utils};
use eframe::egui::{
    self,
    text::{LayoutJob, TextFormat},
};

#[derive(Debug)]
pub struct Task2Tab {
    input_text: String,
    probabilities: Vec<f64>,
    draws: Vec<f64>,
    parse_error: Option<String>,
}

impl Task2Tab {
    pub fn new() -> Self {
        Self {
            input_text: String::from(""),
            probabilities: vec![],
            draws: vec![],
            parse_error: None,
        }
    }
}

impl Tab for Task2Tab {
    fn name(&self) -> &'static str {
        "Task2"
    }

    fn ui(&mut self, ui: &mut egui::Ui) {
        ui.heading(self.name());

        ui.add_space(8.0);

        ui.group(|ui| {
            ui.set_min_width(ui.available_width());
            ui.add(egui::TextEdit::singleline(&mut self.input_text));

            ui.add_space(8.0);

            if ui.button("Compute").clicked() {
                match self
                    .input_text
                    .split(|c: char| c == ',' || c.is_whitespace())
                    .filter(|s| !s.is_empty())
                    .map(|s| utils::parse_in_range(s, 0.0, 1.0))
                    .collect::<Result<Vec<_>, _>>()
                {
                    Ok(vals) => {
                        self.probabilities = vals;
                        self.parse_error = None;
                    }
                    Err(e) => self.parse_error = Some(e.to_string()),
                }

                self.draws.clear();
                for _ in 0..self.probabilities.len() {
                    self.draws.push(rand::random::<f64>());
                }
            }

            if let Some(err) = &self.parse_error {
                ui.add_space(6.0);
                ui.label(egui::RichText::new(err).color(colors::RED));
            }

            ui.separator();

            let font = egui::TextStyle::Body.resolve(ui.style());

            let mut draws_job = LayoutJob::default();
            draws_job.append(
                "Draws: ",
                0.0,
                TextFormat::simple(
                    font.clone(),
                    ui.visuals().weak_text_color(),
                ),
            );

            let mut results_job = LayoutJob::default();
            results_job.append(
                "Result: ",
                0.0,
                TextFormat::simple(font.clone(), ui.visuals().text_color()),
            );

            for (idx, draw) in self.draws.iter().enumerate() {
                let hit = *draw < self.probabilities[idx];
                let (label, color) = if hit {
                    ("True", colors::GREEN)
                } else {
                    ("False", colors::RED)
                };

                draws_job.append(
                    &format!("{draw:.3}"),
                    0.0,
                    TextFormat::simple(
                        font.clone(),
                        ui.visuals().weak_text_color(),
                    ),
                );

                results_job.append(
                    label,
                    0.0,
                    TextFormat::simple(font.clone(), color),
                );
                if idx < self.draws.len() - 1 {
                    draws_job.append(
                        ", ",
                        0.0,
                        TextFormat::simple(
                            font.clone(),
                            ui.visuals().text_color(),
                        ),
                    );
                    results_job.append(
                        ", ",
                        0.0,
                        TextFormat::simple(
                            font.clone(),
                            ui.visuals().text_color(),
                        ),
                    );
                }
            }

            ui.label(draws_job);

            ui.add_space(6.0);

            ui.label(results_job);

            // if let Some(draw) = self.last_draw {
            //     let hit = draw < self.probability;
            //     let (label, color) = if hit {
            //         ("True", egui::Color32::from_rgb(0x2e, 0x7d, 0x32))
            //     } else {
            //         ("False", egui::Color32::from_rgb(0xc6, 0x28, 0x28))
            //     };

            //     ui.label(
            //         egui::RichText::new(format!(
            //             "Draw: {draw:.3} · Probability {:.3}",
            //             self.probability
            //         ))
            //         .weak(),
            //     );

            //     job.append(
            //         "Result: ",
            //         0.0,
            //         TextFormat::simple(font.clone(), ui.visuals().text_color()),
            //     );
            //     job.append(label, 0.0, TextFormat::simple(font, color));

            //     ui.label(job);
            // } else {
            //     ui.label(egui::RichText::new("-").weak());
            //     ui.label(egui::RichText::new(" ").weak());
            // }
        });
    }
}

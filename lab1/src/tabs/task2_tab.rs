use std::vec;

const MAX_EVENTS: usize = 12;

use crate::common::{colors, sampling, stats, tab::Tab, utils};
use eframe::egui::{
    self,
    text::{LayoutJob, TextFormat},
};

#[derive(Debug)]
struct RunStats {
    runs: u32,
    probabilities: Vec<f64>,
    counts: Vec<u64>,
    events: usize,
}

fn combination_probabilities(p: &[f64]) -> Vec<f64> {
    (0..1usize << p.len())
        .map(|mask| {
            p.iter()
                .enumerate()
                .map(|(i, &pi)| if mask >> i & 1 == 1 { pi } else { 1.0 - pi })
                .product()
        })
        .collect()
}

fn combination_label(mask: usize, events: usize) -> String {
    (0..events)
        .map(|i| if mask >> i & 1 == 1 { 'T' } else { 'F' })
        .collect()
}

#[derive(Debug)]
pub struct Task2Tab {
    input_text: String,
    probabilities: Vec<f64>,
    draws: Vec<f64>,
    parse_error: Option<String>,
    runs: u32,
    stats: Option<RunStats>,
}

impl Task2Tab {
    pub fn new() -> Self {
        Self {
            input_text: String::from(""),
            probabilities: vec![],
            draws: vec![],
            parse_error: None,
            runs: stats::DEFAULT_RUNS,
            stats: None,
        }
    }

    fn run_many(&mut self) -> Result<(), String> {
        let p = utils::parse_probability_list(&self.input_text)?;

        if p.is_empty() {
            return Err("Enter at least one probability.".to_owned());
        }
        if p.len() > MAX_EVENTS {
            return Err(format!(
                "Too many events: {} (max {MAX_EVENTS}, {} combinations).",
                p.len(),
                1usize << MAX_EVENTS
            ));
        }

        let mut counts = vec![0u64; 1 << p.len()];

        for _ in 0..self.runs {
            let mut mask = 0usize;
            for (i, &pi) in p.iter().enumerate() {
                if sampling::is_hit(rand::random(), pi) {
                    mask |= 1 << i;
                }
            }
            counts[mask] += 1;
        }

        self.stats = Some(RunStats {
            runs: self.runs,
            probabilities: combination_probabilities(&p),
            counts,
            events: p.len(),
        });

        Ok(())
    }

    fn draw_stats(&self, ui: &mut egui::Ui) {
        let Some(run) = &self.stats else {
            return;
        };

        ui.heading(format!(
            "Statistics ({} runs, {} combinations)",
            run.runs,
            run.counts.len()
        ));

        let labels: Vec<String> = (0..run.counts.len())
            .map(|m| combination_label(m, run.events))
            .collect();
        egui::ScrollArea::vertical()
            .id_salt("task2_stats_scroll")
            .max_height(320.0)
            .show(ui, |ui| {
                stats::show_frequency_report(
                    ui,
                    "task2_stats",
                    run.runs as u64,
                    &labels,
                    &run.probabilities,
                    &run.counts,
                );
            });

        if let Some(test) = stats::chi_square(&run.counts, &run.probabilities) {
            stats::show_chi_square(ui, &test);
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
                match utils::parse_probability_list(&self.input_text) {
                    Ok(vals) => {
                        self.probabilities = vals;
                        self.parse_error = None;
                    }
                    Err(e) => self.parse_error = Some(e),
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

            ui.separator();

            if stats::runs_control(ui, &mut self.runs) {
                self.parse_error = self.run_many().err();
            }

            ui.add_space(8.0);

            self.draw_stats(ui);
        });
    }
}

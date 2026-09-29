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
}

#[derive(Debug)]
pub struct Task4Tab {
    input_text: String,
    probabilities: Vec<f64>,
    cumulative: Vec<f64>,
    draw: Option<f64>,
    result: Option<usize>,
    parse_error: Option<String>,
    runs: u32,
    stats: Option<RunStats>,
}

impl Task4Tab {
    pub fn new() -> Self {
        Self {
            input_text: String::new(),
            probabilities: Vec::new(),
            cumulative: Vec::new(),
            draw: None,
            result: None,
            parse_error: None,
            runs: stats::DEFAULT_RUNS,
            stats: None,
        }
    }

    fn parse_probabilities(input: &str) -> Result<Vec<f64>, String> {
        let probabilities = utils::parse_probability_list(input)?;

        if probabilities.is_empty() {
            return Err("Enter at least one probability.".to_owned());
        }

        let sum: f64 = probabilities.iter().sum();

        if (sum - 1.0).abs() > 1e-10 {
            return Err(format!(
                "The probabilities must sum to 1. Current sum: {sum:.6}"
            ));
        }

        Ok(probabilities)
    }

    fn compute(&mut self) -> Result<(), String> {
        let probabilities = Self::parse_probabilities(&self.input_text)?;

        let cumulative = sampling::cumulative(&probabilities);

        let draw = rand::random::<f64>();

        let result = sampling::pick_index(&cumulative, draw);

        self.probabilities = probabilities;
        self.cumulative = cumulative;
        self.draw = Some(draw);
        self.result = Some(result);

        Ok(())
    }

    fn run_many(&mut self) -> Result<(), String> {
        let probabilities = Self::parse_probabilities(&self.input_text)?;
        let cumulative = sampling::cumulative(&probabilities);

        let mut counts = vec![0u64; probabilities.len()];

        for _ in 0..self.runs {
            counts[sampling::pick_index(&cumulative, rand::random())] += 1;
        }

        self.stats = Some(RunStats {
            runs: self.runs,
            probabilities,
            counts,
        });

        Ok(())
    }

    fn draw_stats(&self, ui: &mut egui::Ui) {
        let Some(run) = &self.stats else {
            return;
        };

        ui.heading(format!("Statistics ({} runs)", run.runs));

        let labels: Vec<String> = (0..run.probabilities.len())
            .map(|index| index.to_string())
            .collect();

        stats::show_frequency_report(
            ui,
            "task4_stats",
            run.runs as u64,
            &labels,
            &run.probabilities,
            &run.counts,
        );

        ui.add_space(8.0);

        if let Some(test) = stats::chi_square(&run.counts, &run.probabilities) {
            stats::show_chi_square(ui, &test);
        }
    }
}

impl Tab for Task4Tab {
    fn name(&self) -> &'static str {
        "Task4"
    }

    fn ui(&mut self, ui: &mut egui::Ui) {
        ui.heading(self.name());

        ui.add_space(8.0);

        ui.group(|ui| {
            ui.set_min_width(ui.available_width());

            ui.horizontal(|ui| {
                ui.label("Probabilities:");

                ui.add(
                    egui::TextEdit::singleline(&mut self.input_text)
                        .hint_text("0.2, 0.5, 0.3")
                        .desired_width(260.0),
                );
            });

            ui.add_space(8.0);

            if ui.button("Compute").clicked() {
                self.probabilities.clear();
                self.cumulative.clear();
                self.draw = None;
                self.result = None;

                match self.compute() {
                    Ok(()) => {
                        self.parse_error = None;
                    }
                    Err(error) => {
                        self.parse_error = Some(error);
                    }
                }
            }

            if let Some(error) = &self.parse_error {
                ui.add_space(6.0);
                ui.label(egui::RichText::new(error).color(colors::RED));
            }

            ui.separator();

            if let (Some(draw), Some(result)) = (self.draw, self.result) {
                let font = egui::TextStyle::Body.resolve(ui.style());

                let mut probabilities_job = LayoutJob::default();

                probabilities_job.append(
                    "Probabilities: ",
                    0.0,
                    TextFormat::simple(font.clone(), ui.visuals().text_color()),
                );

                for (idx, probability) in self.probabilities.iter().enumerate()
                {
                    probabilities_job.append(
                        &format!("{probability:.3}"),
                        0.0,
                        TextFormat::simple(
                            font.clone(),
                            ui.visuals().weak_text_color(),
                        ),
                    );

                    if idx < self.probabilities.len() - 1 {
                        probabilities_job.append(
                            ", ",
                            0.0,
                            TextFormat::simple(
                                font.clone(),
                                ui.visuals().text_color(),
                            ),
                        );
                    }
                }

                ui.label(probabilities_job);

                ui.add_space(6.0);

                let mut cumulative_job = LayoutJob::default();

                cumulative_job.append(
                    "Cumulative: ",
                    0.0,
                    TextFormat::simple(font.clone(), ui.visuals().text_color()),
                );

                for (idx, value) in self.cumulative.iter().enumerate() {
                    cumulative_job.append(
                        &format!("{value:.3}"),
                        0.0,
                        TextFormat::simple(
                            font.clone(),
                            ui.visuals().weak_text_color(),
                        ),
                    );

                    if idx < self.cumulative.len() - 1 {
                        cumulative_job.append(
                            ", ",
                            0.0,
                            TextFormat::simple(
                                font.clone(),
                                ui.visuals().text_color(),
                            ),
                        );
                    }
                }

                ui.label(cumulative_job);

                ui.add_space(6.0);

                let mut draw_job = LayoutJob::default();

                draw_job.append(
                    "Draw: ",
                    0.0,
                    TextFormat::simple(font.clone(), ui.visuals().text_color()),
                );

                draw_job.append(
                    &format!("{draw:.3}"),
                    0.0,
                    TextFormat::simple(
                        font.clone(),
                        ui.visuals().weak_text_color(),
                    ),
                );

                ui.label(draw_job);

                ui.add_space(6.0);

                let mut result_job = LayoutJob::default();

                result_job.append(
                    "Result: ",
                    0.0,
                    TextFormat::simple(font.clone(), ui.visuals().text_color()),
                );

                result_job.append(
                    &format!("{result}"),
                    0.0,
                    TextFormat::simple(font, colors::GREEN),
                );

                ui.label(result_job);
            } else if self.parse_error.is_none() {
                ui.label(
                    egui::RichText::new(
                        "Enter probabilities and press Compute.",
                    )
                    .weak(),
                );
            }

            ui.separator();

            if stats::runs_control(ui, &mut self.runs) {
                self.parse_error = self.run_many().err();
            }

            ui.add_space(8.0);

            self.draw_stats(ui);
        });
    }
}

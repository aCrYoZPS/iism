use crate::common::{colors, tab::Tab, utils};
use eframe::egui::{
    self,
    text::{LayoutJob, TextFormat},
};

#[derive(Debug)]
pub struct Task4Tab {
    input_text: String,
    probabilities: Vec<f64>,
    cumulative: Vec<f64>,
    draw: Option<f64>,
    result: Option<usize>,
    parse_error: Option<String>,
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
        }
    }

    fn parse_probabilities(input: &str) -> Result<Vec<f64>, String> {
        let probabilities = input
            .split(|c: char| c == ',' || c.is_whitespace())
            .filter(|part| !part.is_empty())
            .map(|part| {
                utils::parse_in_range(part, 0.0, 1.0)
                    .map_err(|error| error.to_string())
            })
            .collect::<Result<Vec<f64>, String>>()?;

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

        let mut cumulative = Vec::with_capacity(probabilities.len());
        let mut sum = 0.0;

        for probability in &probabilities {
            sum += probability;
            cumulative.push(sum);
        }

        // Avoid a possible floating-point value slightly below 1.0.
        if let Some(last) = cumulative.last_mut() {
            *last = 1.0;
        }

        let draw = rand::random::<f64>();

        let result = cumulative
            .iter()
            .position(|upper_bound| draw < *upper_bound)
            .unwrap_or(probabilities.len() - 1);

        self.probabilities = probabilities;
        self.cumulative = cumulative;
        self.draw = Some(draw);
        self.result = Some(result);

        Ok(())
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
        });
    }
}

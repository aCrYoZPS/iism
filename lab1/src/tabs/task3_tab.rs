use crate::common::{colors, sampling, stats, tab::Tab, utils};
use eframe::egui::{
    self,
    text::{LayoutJob, TextFormat},
};

#[derive(Debug, Clone, Copy)]
enum Outcome {
    AB,
    ANotB,
    NotAB,
    NotANotB,
}

impl Outcome {
    fn label(self) -> &'static str {
        match self {
            Self::AB => "0",
            Self::ANotB => "1",
            Self::NotAB => "2",
            Self::NotANotB => "3",
        }
    }

    fn from_index(index: usize) -> Self {
        match index {
            0 => Self::AB,
            1 => Self::ANotB,
            2 => Self::NotAB,
            _ => Self::NotANotB,
        }
    }
}

#[derive(Debug)]
struct RunStats {
    runs: u32,
    probabilities: [f64; 4],
    counts: [u64; 4],
}

#[derive(Debug)]
pub struct Task3Tab {
    p_a_text: String,
    p_b_given_a_text: String,
    p_b_text: String,

    probabilities: Option<[f64; 4]>,
    draw: Option<f64>,
    outcome: Option<Outcome>,

    parse_error: Option<String>,

    runs: u32,
    stats: Option<RunStats>,
}

impl Task3Tab {
    pub fn new() -> Self {
        Self {
            p_a_text: String::new(),
            p_b_given_a_text: String::new(),
            p_b_text: String::new(),

            probabilities: None,
            draw: None,
            outcome: None,

            parse_error: None,

            runs: stats::DEFAULT_RUNS,
            stats: None,
        }
    }

    fn parse_probability(text: &str) -> Result<f64, String> {
        utils::parse_in_range(text.trim(), 0.0, 1.0).map_err(|e| e.to_string())
    }

    fn sample(draw: f64, probabilities: [f64; 4]) -> Outcome {
        let cumulative = sampling::cumulative(&probabilities);
        Outcome::from_index(sampling::pick_index(&cumulative, draw))
    }

    fn joint_probabilities(&self) -> Result<[f64; 4], String> {
        let p_a = Self::parse_probability(&self.p_a_text)?;
        let p_b_given_a = Self::parse_probability(&self.p_b_given_a_text)?;
        let p_b = Self::parse_probability(&self.p_b_text)?;

        let p_not_a = 1.0 - p_a;

        let p_ab = p_a * p_b_given_a;

        let p_a_not_b = p_a * (1.0 - p_b_given_a);

        let p_not_a_b = p_b - p_ab;

        let p_not_a_not_b = p_not_a - p_not_a_b;

        if p_not_a_b < 0.0 || p_not_a_not_b < 0.0 {
            return Err(
                "Invalid probabilities: they produce a negative probability. \
                 Check that P(A), P(B | A), and P(B) are compatible."
                    .to_owned(),
            );
        }

        let probabilities = [
            p_ab.max(0.0),
            p_a_not_b.max(0.0),
            p_not_a_b.max(0.0),
            p_not_a_not_b.max(0.0),
        ];

        let total: f64 = probabilities.iter().sum();

        if (total - 1.0).abs() > 1e-9 {
            return Err("Invalid probabilities: the total is not equal to 1."
                .to_owned());
        }

        Ok(probabilities)
    }

    fn compute(&mut self) -> Result<(), String> {
        let probabilities = self.joint_probabilities()?;

        let draw = rand::random::<f64>();

        self.probabilities = Some(probabilities);
        self.draw = Some(draw);
        self.outcome = Some(Self::sample(draw, probabilities));

        Ok(())
    }

    fn run_many(&mut self) -> Result<(), String> {
        let probabilities = self.joint_probabilities()?;
        let cumulative = sampling::cumulative(&probabilities);

        let mut counts = [0u64; 4];

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

        let labels: Vec<String> = [
            "0 (A, B)",
            "1 (A, \u{ac}B)",
            "2 (\u{ac}A, B)",
            "3 (\u{ac}A, \u{ac}B)",
        ]
        .map(str::to_owned)
        .to_vec();

        stats::show_frequency_report(
            ui,
            "task3_stats",
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

impl Tab for Task3Tab {
    fn name(&self) -> &'static str {
        "Task3"
    }

    fn ui(&mut self, ui: &mut egui::Ui) {
        egui::ScrollArea::vertical()
            .auto_shrink([false, false])
            .show(ui, |ui| {
                ui.heading(self.name());

                ui.add_space(8.0);

                ui.group(|ui| {
                    ui.set_min_width(ui.available_width());

                    ui.horizontal(|ui| {
                        ui.label("P(A):");
                        ui.add(
                            egui::TextEdit::singleline(&mut self.p_a_text)
                                .hint_text("0.0 .. 1.0")
                                .desired_width(180.0),
                        );
                    });

                    ui.horizontal(|ui| {
                        ui.label("P(B | A):");
                        ui.add(
                            egui::TextEdit::singleline(
                                &mut self.p_b_given_a_text,
                            )
                            .hint_text("0.0 .. 1.0")
                            .desired_width(180.0),
                        );
                    });

                    ui.horizontal(|ui| {
                        ui.label("P(B):");
                        ui.add(
                            egui::TextEdit::singleline(&mut self.p_b_text)
                                .hint_text("0.0 .. 1.0")
                                .desired_width(180.0),
                        );
                    });

                    ui.add_space(8.0);

                    if ui.button("Compute").clicked() {
                        self.probabilities = None;
                        self.draw = None;
                        self.outcome = None;

                        match self.compute() {
                            Ok(()) => self.parse_error = None,
                            Err(error) => self.parse_error = Some(error),
                        }
                    }

                    if let Some(error) = &self.parse_error {
                        ui.add_space(6.0);
                        ui.label(egui::RichText::new(error).color(colors::RED));
                    }

                    ui.separator();

                    let font = egui::TextStyle::Body.resolve(ui.style());

                    if let Some([p_ab, p_a_not_b, p_not_a_b, p_not_a_not_b]) =
                        self.probabilities
                    {
                        ui.label(format!("P(A, B) = {p_ab:.4}"));
                        ui.label(format!("P(A, ¬B) = {p_a_not_b:.4}"));
                        ui.label(format!("P(¬A, B) = {p_not_a_b:.4}"));
                        ui.label(format!("P(¬A, ¬B) = {p_not_a_not_b:.4}"));

                        ui.add_space(8.0);

                        let draw =
                            self.draw.expect("draw is set with probabilities");
                        let outcome = self
                            .outcome
                            .expect("outcome is set with probabilities");

                        let mut draw_job = LayoutJob::default();
                        draw_job.append(
                            "Draw: ",
                            0.0,
                            TextFormat::simple(
                                font.clone(),
                                ui.visuals().weak_text_color(),
                            ),
                        );
                        draw_job.append(
                            &format!("{draw:.4}"),
                            0.0,
                            TextFormat::simple(
                                font.clone(),
                                ui.visuals().weak_text_color(),
                            ),
                        );
                        ui.label(draw_job);

                        let mut result_job = LayoutJob::default();
                        result_job.append(
                            "Result: ",
                            0.0,
                            TextFormat::simple(
                                font.clone(),
                                ui.visuals().text_color(),
                            ),
                        );
                        result_job.append(
                            outcome.label(),
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
            });
    }
}

use crate::common::tab::Tab;
use eframe::egui;
use rand;

#[derive(Debug)]
pub struct Task1Tab {
    probability: f64,
    text: Option<String>,
}

impl Task1Tab {
    pub fn new() -> Self {
        Self {
            probability: 0.5,
            text: None,
        }
    }
}

impl Tab for Task1Tab {
    fn name(&self) -> &'static str {
        "Task1"
    }

    fn ui(&mut self, ui: &mut egui::Ui) {
        ui.heading("Task1");

        ui.add(
            egui::Slider::new(&mut self.probability, 0.0..=1.0)
                .text("Probability (0–1)"),
        );

        if ui.button("Compute").clicked() {
            let border: f64 = rand::random();
            if border <= self.probability {
                self.text = Some(String::from("True"));
            } else {
                self.text = Some(String::from("False"));
            }
        }

        ui.label(match &self.text {
            Some(txt) => txt,
            None => "",
        });
    }
}

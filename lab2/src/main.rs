mod analysis;
mod continuous;
mod discrete;
mod discrete_params;
mod params;
mod report;

use common::tab::Tab;
use continuous::ContinuousTab;
use discrete::DiscreteTab;
use eframe::egui;

struct LabApp {
    selected_tab: usize,
    tabs: Vec<Box<dyn Tab>>,
}

impl LabApp {
    fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        let tabs: Vec<Box<dyn Tab>> =
            vec![Box::new(ContinuousTab::new()), Box::new(DiscreteTab::new())];

        Self {
            selected_tab: 0,
            tabs,
        }
    }
}

impl eframe::App for LabApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::TopBottomPanel::top("tabs").show(ctx, |ui| {
            ui.horizontal(|ui| {
                for (i, tab) in self.tabs.iter().enumerate() {
                    ui.selectable_value(&mut self.selected_tab, i, tab.name());
                }
            });
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            if let Some(tab) = self.tabs.get_mut(self.selected_tab) {
                tab.ui(ui);
            }
        });
    }
}

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions::default();
    eframe::run_native(
        "Lab2",
        options,
        Box::new(|cc| Ok(Box::new(LabApp::new(cc)))),
    )
}

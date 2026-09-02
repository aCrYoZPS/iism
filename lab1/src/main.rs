mod common;
mod tabs;

use crate::tabs::task1_tab::*;
use common::tab::Tab;
use eframe::egui;

struct LabApp {
    selected_tab: usize,
    tabs: Vec<Box<dyn Tab>>,
}

impl LabApp {
    fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        let tabs: Vec<Box<dyn Tab>> = vec![Box::new(Task1Tab::new())];

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

        // Content
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
        "Lab Multi-Tab App",
        options,
        Box::new(|cc| Ok(Box::new(LabApp::new(cc)))),
    )
}

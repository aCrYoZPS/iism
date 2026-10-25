mod chain;
mod task;
mod task_tab;
mod tasks;

use common::tab::Tab;
use eframe::egui;
use task_tab::TaskTab;

struct LabApp {
    selected_tab: usize,
    tabs: Vec<Box<dyn Tab>>,
}

impl LabApp {
    fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        let tabs: Vec<Box<dyn Tab>> = vec![
            Box::new(TaskTab::new(tasks::queue::task())),
            Box::new(TaskTab::new(tasks::machines::task())),
            Box::new(TaskTab::new(tasks::breakdown::task())),
            Box::new(TaskTab::new(tasks::priority::task())),
            Box::new(TaskTab::new(tasks::phases::task())),
            Box::new(TaskTab::new(tasks::shop::task())),
        ];

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
        "Lab4",
        options,
        Box::new(|cc| Ok(Box::new(LabApp::new(cc)))),
    )
}

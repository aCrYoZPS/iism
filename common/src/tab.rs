use eframe::egui;

pub trait Tab: Send + 'static {
    fn name(&self) -> &'static str;
    fn ui(&mut self, ui: &mut egui::Ui);
}

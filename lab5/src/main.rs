mod app;
mod canvas;

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions::default();
    eframe::run_native(
        "Lab5",
        options,
        Box::new(|_cc| Ok(Box::new(app::App::new()))),
    )
}

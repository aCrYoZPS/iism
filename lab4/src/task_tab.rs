use common::{colors, tab::Tab};
use eframe::egui;
use egui_plot::{Bar, BarChart, Plot};

use crate::task::{Kind, Param, Solution, Task};

const CHART_HEIGHT: f32 = 280.0;
const TABLE_HEIGHT: f32 = 240.0;

pub struct TaskTab {
    task: Task,
    result: Result<Solution, String>,
}

impl TaskTab {
    pub fn new(task: Task) -> Self {
        let mut tab = Self {
            result: Err(String::new()),
            task,
        };
        tab.recompute();
        tab
    }

    fn recompute(&mut self) {
        let values: Vec<f64> =
            self.task.params.iter().map(|param| param.value).collect();
        self.result = (self.task.solve)(&values);
    }
}

fn edit_param(ui: &mut egui::Ui, param: &mut Param) -> bool {
    match param.kind {
        Kind::Real => {
            ui.label(param.name);
            ui.add(
                egui::DragValue::new(&mut param.value)
                    .speed(0.01)
                    .range(0.0..=f64::INFINITY),
            )
            .changed()
        }
        Kind::Count { min } => {
            let mut value = param.value as u32;
            ui.label(param.name);
            let changed = ui
                .add(egui::DragValue::new(&mut value).range(min..=1000))
                .changed();
            param.value = value as f64;
            changed
        }
        Kind::Flag => {
            let mut on = param.value > 0.5;
            let changed = ui.checkbox(&mut on, param.name).changed();
            param.value = if on { 1.0 } else { 0.0 };
            changed
        }
    }
}

fn show_distribution(ui: &mut egui::Ui, id: &str, solution: &Solution) {
    let bars = solution
        .p
        .iter()
        .enumerate()
        .map(|(k, &pk)| Bar::new(k as f64, pk).width(0.7))
        .collect();
    let labels = solution.labels.clone();

    Plot::new(id)
        .height(CHART_HEIGHT)
        .include_y(0.0)
        .allow_zoom(false)
        .allow_drag(false)
        .allow_scroll(false)
        .allow_boxed_zoom(false)
        .x_axis_formatter(move |mark, _| {
            let index = mark.value.round();
            if (mark.value - index).abs() < 1e-6 && index >= 0.0 {
                labels.get(index as usize).cloned().unwrap_or_default()
            } else {
                String::new()
            }
        })
        .show(ui, |plot_ui| {
            plot_ui.bar_chart(BarChart::new("p", bars).color(colors::BLUE));
        });
}

fn show_solution(ui: &mut egui::Ui, id: &str, solution: &Solution) {
    for warning in &solution.warnings {
        ui.label(egui::RichText::new(warning).color(colors::RED));
    }

    // network tasks have no single state distribution to show
    if !solution.p.is_empty() {
        ui.heading("Final probabilities");
        show_distribution(ui, &format!("{id}_p"), solution);

        egui::ScrollArea::vertical()
            .id_salt(format!("{id}_states"))
            .max_height(TABLE_HEIGHT)
            .show(ui, |ui| {
                egui::Grid::new(format!("{id}_states_grid")).show(ui, |ui| {
                    for (label, p) in solution.labels.iter().zip(&solution.p) {
                        ui.label(label);
                        ui.label(format!("{p:.6}"));
                        ui.end_row();
                    }
                });
            });

        ui.add_space(8.0);
    }

    ui.heading("Efficiency");
    egui::Grid::new(format!("{id}_metrics")).show(ui, |ui| {
        for metric in &solution.metrics {
            ui.label(metric.name);
            ui.label(format!("{:.6}", metric.value));
            ui.end_row();
        }
    });
}

impl Tab for TaskTab {
    fn name(&self) -> &'static str {
        self.task.name
    }

    fn ui(&mut self, ui: &mut egui::Ui) {
        ui.heading(self.name());

        ui.add_space(8.0);

        let mut changed = false;
        ui.horizontal_wrapped(|ui| {
            for param in &mut self.task.params {
                changed |= edit_param(ui, param);
            }
        });
        if changed {
            self.recompute();
        }

        ui.add_space(8.0);

        let id = self.task.name;
        match &self.result {
            Ok(solution) => {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    show_solution(ui, id, solution);
                });
            }
            Err(message) => {
                ui.label(egui::RichText::new(message).color(colors::RED));
            }
        }
    }
}

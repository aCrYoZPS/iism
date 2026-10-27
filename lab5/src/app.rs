use common::colors;
use eframe::egui;
use lab5::analysis::{self, NetClass, Properties};
use lab5::graph::ReachabilityGraph;
use lab5::net::{Marking, Net};
use lab5::nets;
use lab5::parse::{parse, parse_marking};

use crate::canvas::{self, Layout};

const VERTEX_LIMIT: usize = 10_000;
const SHOWN_PAIRS: usize = 10;
const SHOWN_DEADLOCKS: usize = 5;

/// A successfully parsed net with everything computed from it.
struct Loaded {
    net: Net,
    initial: Marking,
    graph: ReachabilityGraph,
    class: NetClass,
    props: Properties,
    pairs: Vec<(usize, usize, usize)>,
    layout: Layout,
    marking: Marking,
    /// Markings before each step, for "Back".
    history: Vec<Marking>,
    /// Text of each marking graph vertex, see `format_marking`.
    labels: Vec<String>,
}

impl Loaded {
    fn new(net: Net, initial: Marking) -> Self {
        let graph =
            ReachabilityGraph::build(&net, initial.clone(), VERTEX_LIMIT);
        let mut loaded = Self {
            class: analysis::classify(&net),
            props: analysis::analyze(&graph, &net),
            pairs: graph.concurrent_pairs(&net),
            layout: Layout::circle(net.places.len(), net.transitions.len()),
            marking: initial.clone(),
            history: Vec::new(),
            labels: Vec::new(),
            graph,
            initial,
            net,
        };
        loaded.labels = loaded
            .graph
            .markings
            .iter()
            .map(|m| loaded.format_marking(m))
            .collect();
        loaded
    }

    fn fire(&mut self, transition: usize) {
        if let Some(next) = self.net.fire(&self.marking, transition) {
            let previous = std::mem::replace(&mut self.marking, next);
            self.history.push(previous);
        }
    }

    fn back(&mut self) {
        if let Some(previous) = self.history.pop() {
            self.marking = previous;
        }
    }

    fn reset(&mut self) {
        self.marking = self.initial.clone();
        self.history.clear();
    }

    /// Non-empty places only, e.g. `p1=1 p6=2`.
    fn format_marking(&self, marking: &Marking) -> String {
        let parts: Vec<_> = self
            .net
            .places
            .iter()
            .zip(marking)
            .filter(|&(_, &count)| count > 0)
            .map(|(name, count)| format!("{name}={count}"))
            .collect();
        if parts.is_empty() {
            "(empty)".to_string()
        } else {
            parts.join(" ")
        }
    }
}

#[derive(Clone, Copy, PartialEq)]
enum View {
    Net,
    MarkingGraph,
}

pub struct App {
    view: View,
    text: String,
    /// Kept when the text stops parsing, so the picture does not vanish while
    /// the user is typing.
    loaded: Option<Loaded>,
    errors: Vec<String>,
    target: String,
}

impl App {
    pub fn new() -> Self {
        let mut app = Self {
            view: View::Net,
            text: nets::RIGHT_TEXT.to_string(),
            loaded: None,
            errors: Vec::new(),
            target: String::new(),
        };
        app.reload();
        app
    }

    fn reload(&mut self) {
        match parse(&self.text) {
            Ok((net, marking)) => {
                self.loaded = Some(Loaded::new(net, marking));
                self.errors.clear();
            }
            Err(errors) => {
                self.errors = errors.iter().map(|e| e.to_string()).collect();
            }
        }
    }
}

fn yes_no(value: bool) -> &'static str {
    if value { "yes" } else { "no" }
}

fn class_name(class: &NetClass) -> &'static str {
    match (class.state_machine, class.marked_graph, class.free_choice) {
        (true, true, _) => "state machine and marked graph",
        (true, false, _) => "state machine",
        (false, true, _) => "marked graph",
        (false, false, true) => "free-choice net",
        (false, false, false) => "general net",
    }
}

fn show_simulation(ui: &mut egui::Ui, loaded: &mut Loaded) {
    ui.heading("Simulation");
    ui.horizontal(|ui| {
        if ui
            .add_enabled(!loaded.history.is_empty(), egui::Button::new("Back"))
            .clicked()
        {
            loaded.back();
        }
        if ui.button("Reset").clicked() {
            loaded.reset();
        }
    });
    ui.label(format!(
        "Marking: {}",
        loaded.format_marking(&loaded.marking)
    ));
    let enabled: Vec<_> = loaded
        .net
        .enabled_transitions(&loaded.marking)
        .into_iter()
        .map(|t| loaded.net.transitions[t].name.as_str())
        .collect();
    if enabled.is_empty() {
        ui.label(
            egui::RichText::new("Deadlock: no transition is enabled")
                .color(colors::RED),
        );
    } else {
        ui.label(format!("Enabled: {}", enabled.join(", ")));
    }
    ui.label("Click a green transition to fire it. Drag nodes to move them.");
}

fn show_analysis(ui: &mut egui::Ui, loaded: &Loaded, target: &mut String) {
    let Loaded {
        net,
        graph,
        class,
        props,
        pairs,
        ..
    } = loaded;

    ui.heading("Class");
    ui.label(class_name(class));
    ui.label(format!(
        "state machine: {}, marked graph: {}, free-choice: {}",
        yes_no(class.state_machine),
        yes_no(class.marked_graph),
        yes_no(class.free_choice),
    ));

    ui.heading("Reachability graph");
    ui.label(format!(
        "{} markings, {} edges",
        graph.markings.len(),
        graph.edges.len()
    ));
    if !props.complete {
        ui.label(
            egui::RichText::new(format!(
                "Search stopped at {VERTEX_LIMIT} markings: the net may be \
                 unbounded. Everything below describes only the explored \
                 part."
            ))
            .color(colors::RED),
        );
    }

    ui.heading("Properties");
    let bounds: Vec<_> = net
        .places
        .iter()
        .zip(&props.bounds)
        .map(|(name, bound)| format!("{name}: {bound}"))
        .collect();
    ui.label(format!("Bounds: {}", bounds.join(", ")));
    ui.label(format!(
        "Safe (bound 1): {}",
        yes_no(props.bounds.iter().all(|&b| b <= 1))
    ));
    ui.label(format!("Reversible: {}", yes_no(props.reversible)));
    ui.label(format!("Live: {}", yes_no(props.live)));
    if props.deadlocks.is_empty() {
        ui.label("Deadlocks: none");
    } else {
        ui.label(format!("Deadlocks: {}", props.deadlocks.len()));
        for &v in props.deadlocks.iter().take(SHOWN_DEADLOCKS) {
            ui.label(format!(
                "  {}",
                loaded.format_marking(&graph.markings[v])
            ));
        }
    }

    ui.heading("Parallel firing");
    if pairs.is_empty() {
        ui.label("Not possible in any reachable marking");
    } else {
        ui.label(format!("Possible, {} cases", pairs.len()));
        for &(v, t, u) in pairs.iter().take(SHOWN_PAIRS) {
            ui.label(format!(
                "  {}: {} and {}",
                loaded.format_marking(&graph.markings[v]),
                net.transitions[t].name,
                net.transitions[u].name,
            ));
        }
        if pairs.len() > SHOWN_PAIRS {
            ui.label(format!("  ... {} more", pairs.len() - SHOWN_PAIRS));
        }
    }

    ui.heading("Is a marking reachable?");
    ui.add(
        egui::TextEdit::singleline(target)
            .hint_text("e.g. p1=1 p6=1")
            .desired_width(f32::INFINITY),
    );
    match parse_marking(target, net) {
        Err(message) => {
            ui.label(egui::RichText::new(message).color(colors::RED));
        }
        Ok(_) if target.trim().is_empty() => {}
        Ok(marking) => match graph.index_of(&marking) {
            Some(v) => {
                ui.label(format!("Reachable (vertex {v})"));
            }
            None if props.complete => {
                ui.label("Not reachable");
            }
            None => {
                ui.label("Not found in the explored part of the graph");
            }
        },
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::SidePanel::left("side")
            .default_width(380.0)
            .show(ctx, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    let mut changed = false;
                    ui.horizontal(|ui| {
                        if ui.button("Load left net").clicked() {
                            self.text = nets::LEFT_TEXT.to_string();
                            changed = true;
                        }
                        if ui.button("Load right net").clicked() {
                            self.text = nets::RIGHT_TEXT.to_string();
                            changed = true;
                        }
                    });
                    changed |= ui
                        .add(
                            egui::TextEdit::multiline(&mut self.text)
                                .code_editor()
                                .desired_rows(12)
                                .desired_width(f32::INFINITY),
                        )
                        .changed();
                    if changed {
                        self.reload();
                    }

                    for error in &self.errors {
                        ui.label(egui::RichText::new(error).color(colors::RED));
                    }
                    if !self.errors.is_empty() {
                        ui.label("Showing the last valid net.");
                    }

                    if let Some(loaded) = &mut self.loaded {
                        show_simulation(ui, loaded);
                        show_analysis(ui, loaded, &mut self.target);
                    }
                });
            });

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.selectable_value(&mut self.view, View::Net, "Net");
                ui.selectable_value(
                    &mut self.view,
                    View::MarkingGraph,
                    "Marking graph",
                );
            });
            let Some(loaded) = &mut self.loaded else {
                return;
            };
            match self.view {
                View::Net => {
                    let clicked = canvas::show(
                        ui,
                        &loaded.net,
                        &loaded.marking,
                        &mut loaded.layout,
                    );
                    if let Some(transition) = clicked {
                        loaded.fire(transition);
                    }
                }
                View::MarkingGraph => {
                    let vertices = loaded.graph.markings.len();
                    if vertices > canvas::MAX_GRAPH_VERTICES {
                        ui.label(format!(
                            "The marking graph has {vertices} vertices; it is \
                             drawn only up to {}.",
                            canvas::MAX_GRAPH_VERTICES
                        ));
                    } else {
                        canvas::show_graph(
                            ui,
                            &loaded.net,
                            &loaded.graph,
                            &loaded.labels,
                            loaded.graph.index_of(&loaded.marking),
                        );
                    }
                }
            }
        });
    }
}

use std::collections::{HashMap, VecDeque};
use std::f32::consts::{FRAC_PI_2, TAU};

use common::colors;
use eframe::egui::{
    self, Align2, Color32, FontId, Painter, Pos2, Rect, Sense, Shape, Stroke,
    StrokeKind, Vec2,
};
use lab5::graph::{Edge, ReachabilityGraph};
use lab5::net::{Marking, Net};

/// Larger marking graphs are not drawn: they turn into an unreadable mess.
pub const MAX_GRAPH_VERTICES: usize = 15;

const PLACE_RADIUS: f32 = 22.0;
const BAR_HALF: Vec2 = Vec2::new(5.0, 18.0);
const HIT_PADDING: f32 = 4.0;
const ARROW_LEN: f32 = 10.0;
const ARROW_HALF_WIDTH: f32 = 5.0;
/// Separation of the two arcs between a place and a transition that are
/// connected in both directions.
const PARALLEL_OFFSET: f32 = 5.0;
const MARGIN: f32 = 50.0;
const FONT_SIZE: f32 = 14.0;
const SMALL_FONT_SIZE: f32 = 12.0;

const NODE_HEIGHT: f32 = 42.0;
const NODE_PADDING: f32 = 16.0;
const GRAPH_MARGIN: Vec2 = Vec2::new(90.0, 50.0);
/// Separation of several edges between the same two vertices.
const LANE_SPACING: f32 = 12.0;
const LOOP_RADIUS: f32 = 12.0;
/// Edges that go back to an earlier layer are drawn as arcs under the
/// picture, one lower arc per edge so they do not overlap.
const BACK_EDGE_CLEARANCE: f32 = 30.0;
const BACK_EDGE_STEP: f32 = 18.0;
const CURVE_SEGMENTS: usize = 24;

/// Node positions in unit space: the canvas maps [-1, 1] to its size, so the
/// picture survives window resizing.
pub struct Layout {
    pub places: Vec<Pos2>,
    pub transitions: Vec<Pos2>,
}

impl Layout {
    /// Places on the outer circle, transitions on a smaller inner one.
    pub fn circle(places: usize, transitions: usize) -> Self {
        let transitions = if transitions == 1 {
            vec![Pos2::ZERO]
        } else {
            ring(transitions, 0.5)
        };
        Self {
            places: ring(places, 1.0),
            transitions,
        }
    }
}

fn ring(count: usize, radius: f32) -> Vec<Pos2> {
    (0..count)
        .map(|i| {
            let angle = TAU * i as f32 / count as f32 - FRAC_PI_2;
            Pos2::new(radius * angle.cos(), radius * angle.sin())
        })
        .collect()
}

/// Draws the net and returns the transition the user clicked, if it is
/// enabled. Nodes can be dragged, which updates `layout`.
pub fn show(
    ui: &mut egui::Ui,
    net: &Net,
    marking: &Marking,
    layout: &mut Layout,
) -> Option<usize> {
    let (area, painter) =
        ui.allocate_painter(ui.available_size(), Sense::hover());
    let center = area.rect.center();
    let scale = (area.rect.size().min_elem() / 2.0 - MARGIN).max(1.0);
    let to_screen = |p: Pos2| center + p.to_vec2() * scale;

    for (i, pos) in layout.places.iter_mut().enumerate() {
        let rect = Rect::from_center_size(
            to_screen(*pos),
            Vec2::splat(2.0 * PLACE_RADIUS),
        );
        let response =
            ui.interact(rect, ui.id().with(("place", i)), Sense::drag());
        *pos += response.drag_delta() / scale;
    }
    let mut clicked = None;
    for (i, pos) in layout.transitions.iter_mut().enumerate() {
        let hit = (BAR_HALF + Vec2::splat(HIT_PADDING)) * 2.0;
        let rect = Rect::from_center_size(to_screen(*pos), hit);
        let response = ui.interact(
            rect,
            ui.id().with(("transition", i)),
            Sense::click_and_drag(),
        );
        *pos += response.drag_delta() / scale;
        if response.clicked() && net.is_enabled(marking, i) {
            clicked = Some(i);
        }
    }

    let visuals = ui.visuals();
    let arc_color = visuals.weak_text_color();
    let text_color = visuals.text_color();
    let fill = visuals.extreme_bg_color;
    let font = FontId::proportional(FONT_SIZE);

    for (i, t) in net.transitions.iter().enumerate() {
        let bar = to_screen(layout.transitions[i]);
        for (places, into_bar) in [(&t.source, true), (&t.target, false)] {
            for &p in places {
                let place = to_screen(layout.places[p]);
                let both = t.source.contains(&p) && t.target.contains(&p);
                let side = if both { PARALLEL_OFFSET } else { 0.0 };
                let shift = perp(bar - place) * side;
                let a = place_edge(place, bar);
                let b = bar_edge(bar, place);
                let (from, to) = if into_bar {
                    (a + shift, b + shift)
                } else {
                    (b - shift, a - shift)
                };
                arrow(&painter, from, to, arc_color);
            }
        }
    }

    for (i, name) in net.places.iter().enumerate() {
        let c = to_screen(layout.places[i]);
        painter.circle(c, PLACE_RADIUS, fill, Stroke::new(2.0, text_color));
        if marking[i] > 0 {
            painter.text(
                c,
                Align2::CENTER_CENTER,
                marking[i].to_string(),
                font.clone(),
                text_color,
            );
        }
        painter.text(
            c - Vec2::new(0.0, PLACE_RADIUS + 2.0),
            Align2::CENTER_BOTTOM,
            name,
            font.clone(),
            text_color,
        );
    }

    for (i, t) in net.transitions.iter().enumerate() {
        let c = to_screen(layout.transitions[i]);
        let color = if net.is_enabled(marking, i) {
            colors::GREEN
        } else {
            arc_color
        };
        painter.rect_filled(
            Rect::from_center_size(c, BAR_HALF * 2.0),
            0.0,
            color,
        );
        painter.text(
            c - Vec2::new(0.0, BAR_HALF.y + 2.0),
            Align2::CENTER_BOTTOM,
            &t.name,
            font.clone(),
            text_color,
        );
    }

    clicked
}

fn perp(v: Vec2) -> Vec2 {
    let v = v.normalized();
    Vec2::new(-v.y, v.x)
}

/// Where the line from a place centre towards `toward` leaves the circle.
fn place_edge(center: Pos2, toward: Pos2) -> Pos2 {
    center + (toward - center).normalized() * PLACE_RADIUS
}

/// Where the line from a bar centre towards `toward` leaves the bar.
fn bar_edge(center: Pos2, toward: Pos2) -> Pos2 {
    rect_edge(center, BAR_HALF, toward)
}

/// Where the line from the centre of a rectangle with half sizes `half`
/// towards `toward` leaves the rectangle.
fn rect_edge(center: Pos2, half: Vec2, toward: Pos2) -> Pos2 {
    let d = toward - center;
    if d == Vec2::ZERO {
        return center;
    }
    let tx = if d.x != 0.0 {
        half.x / d.x.abs()
    } else {
        f32::INFINITY
    };
    let ty = if d.y != 0.0 {
        half.y / d.y.abs()
    } else {
        f32::INFINITY
    };
    center + d * tx.min(ty)
}

/// Layer of each vertex: the length of the shortest path from vertex 0.
pub fn layers(vertices: usize, edges: &[Edge]) -> Vec<usize> {
    let mut layer = vec![usize::MAX; vertices];
    let mut queue = VecDeque::from([0]);
    layer[0] = 0;
    while let Some(v) = queue.pop_front() {
        for e in edges.iter().filter(|e| e.from == v) {
            if layer[e.to] == usize::MAX {
                layer[e.to] = layer[v] + 1;
                queue.push_back(e.to);
            }
        }
    }
    layer
        .into_iter()
        .map(|l| if l == usize::MAX { 0 } else { l })
        .collect()
}

/// Positions in unit space ([-1, 1] both ways): one column per layer, the
/// vertices of a column stacked in index order with a common spacing.
pub fn layered_positions(layers: &[usize]) -> Vec<Pos2> {
    let layer_count = layers.iter().max().map_or(0, |m| m + 1);
    let mut columns = vec![Vec::new(); layer_count];
    for (v, &l) in layers.iter().enumerate() {
        columns[l].push(v);
    }
    let tallest = columns.iter().map(Vec::len).max().unwrap_or(1);
    let step = if tallest > 1 {
        2.0 / (tallest - 1) as f32
    } else {
        0.0
    };

    let mut positions = vec![Pos2::ZERO; layers.len()];
    for (l, column) in columns.iter().enumerate() {
        let x = if layer_count > 1 {
            -1.0 + 2.0 * l as f32 / (layer_count - 1) as f32
        } else {
            0.0
        };
        for (i, &v) in column.iter().enumerate() {
            let y = (i as f32 - (column.len() - 1) as f32 / 2.0) * step;
            positions[v] = Pos2::new(x, y);
        }
    }
    positions
}

/// Draws the marking graph by layers. `labels[v]` is the text of vertex `v`;
/// the `current` vertex is highlighted.
pub fn show_graph(
    ui: &mut egui::Ui,
    net: &Net,
    graph: &ReachabilityGraph,
    labels: &[String],
    current: Option<usize>,
) {
    let (area, painter) =
        ui.allocate_painter(ui.available_size(), Sense::hover());
    let visuals = ui.visuals();
    let edge_color = visuals.weak_text_color();
    let text_color = visuals.text_color();
    let fill = visuals.extreme_bg_color;
    let title = FontId::proportional(FONT_SIZE);
    let small = FontId::proportional(SMALL_FONT_SIZE);

    let layer = layers(graph.markings.len(), &graph.edges);
    let is_back = |e: &Edge| e.from != e.to && layer[e.to] <= layer[e.from];
    let back_count = graph.edges.iter().filter(|e| is_back(e)).count();
    let reserve = if back_count > 0 {
        BACK_EDGE_CLEARANCE + BACK_EDGE_STEP * back_count as f32
    } else {
        0.0
    };

    let half_w = (area.rect.width() / 2.0 - GRAPH_MARGIN.x).max(1.0);
    let usable_h =
        (area.rect.height() - 2.0 * GRAPH_MARGIN.y - reserve).max(2.0);
    let center_y = area.rect.top() + GRAPH_MARGIN.y + usable_h / 2.0;
    let pos: Vec<Pos2> = layered_positions(&layer)
        .into_iter()
        .map(|p| {
            Pos2::new(
                area.rect.center().x + p.x * half_w,
                center_y + p.y * usable_h / 2.0,
            )
        })
        .collect();
    let half: Vec<Vec2> = labels
        .iter()
        .enumerate()
        .map(|(v, label)| {
            let name = painter.layout_no_wrap(
                format!("v{v}"),
                title.clone(),
                text_color,
            );
            let marking = painter.layout_no_wrap(
                label.clone(),
                small.clone(),
                text_color,
            );
            let width = name.size().x.max(marking.size().x) + NODE_PADDING;
            Vec2::new(width / 2.0, NODE_HEIGHT / 2.0)
        })
        .collect();

    // Edges between the same two vertices share a bundle and are spread
    // sideways; back edges and loops are handled separately.
    let mut bundles: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
    for (i, e) in graph.edges.iter().enumerate() {
        if e.from != e.to && !is_back(e) {
            bundles
                .entry((e.from.min(e.to), e.from.max(e.to)))
                .or_default()
                .push(i);
        }
    }
    let mut loops_at = vec![0usize; pos.len()];
    let mut back_index = 0;

    for (i, e) in graph.edges.iter().enumerate() {
        let name = &net.transitions[e.transition].name;
        let label_font = FontId::proportional(SMALL_FONT_SIZE + 1.0);
        let (a, b) = (pos[e.from], pos[e.to]);

        if e.from == e.to {
            let slot = loops_at[e.from];
            loops_at[e.from] += 1;
            let c = a + Vec2::new(
                slot as f32 * (2.0 * LOOP_RADIUS + 10.0),
                -(half[e.from].y + LOOP_RADIUS),
            );
            painter.circle_stroke(c, LOOP_RADIUS, Stroke::new(1.5, edge_color));
            painter.text(
                c - Vec2::new(0.0, LOOP_RADIUS + 1.0),
                Align2::CENTER_BOTTOM,
                name,
                label_font,
                colors::BLUE,
            );
        } else if is_back(e) {
            let y_arc =
                area.rect.bottom() - 15.0 - BACK_EDGE_STEP * back_index as f32;
            back_index += 1;
            let control =
                Pos2::new((a.x + b.x) / 2.0, 2.0 * y_arc - 0.5 * (a.y + b.y));
            let start = rect_edge(a, half[e.from], control);
            let end = rect_edge(b, half[e.to], control);
            let at = |t: f32| {
                let u = 1.0 - t;
                (start.to_vec2() * (u * u)
                    + control.to_vec2() * (2.0 * u * t)
                    + end.to_vec2() * (t * t))
                    .to_pos2()
            };
            let points: Vec<Pos2> = (0..=CURVE_SEGMENTS)
                .map(|k| at(k as f32 / CURVE_SEGMENTS as f32))
                .collect();
            let last = points.len() - 1;
            painter.add(Shape::line(
                points[..last].to_vec(),
                Stroke::new(1.5, edge_color),
            ));
            arrow(&painter, points[last - 1], points[last], edge_color);
            painter.text(
                at(0.5) - Vec2::new(0.0, 2.0),
                Align2::CENTER_BOTTOM,
                name,
                label_font,
                colors::BLUE,
            );
        } else {
            let bundle = &bundles[&(e.from.min(e.to), e.from.max(e.to))];
            let lane = bundle.iter().position(|&j| j == i).unwrap();
            let spread = lane as f32 - (bundle.len() - 1) as f32 / 2.0;
            let (lo, hi) = if e.from < e.to { (a, b) } else { (b, a) };
            let shift = perp(hi - lo) * spread * LANE_SPACING;
            let start = rect_edge(a, half[e.from], b) + shift;
            let end = rect_edge(b, half[e.to], a) + shift;
            arrow(&painter, start, end, edge_color);
            painter.text(
                start.lerp(end, 0.5) - Vec2::new(0.0, 2.0),
                Align2::CENTER_BOTTOM,
                name,
                label_font,
                colors::BLUE,
            );
        }
    }

    for (v, label) in labels.iter().enumerate() {
        let rect = Rect::from_center_size(pos[v], half[v] * 2.0);
        let is_current = current == Some(v);
        let (node_fill, stroke) = if is_current {
            (
                colors::ORANGE.gamma_multiply(0.25),
                Stroke::new(3.0, colors::ORANGE),
            )
        } else {
            (fill, Stroke::new(2.0, text_color))
        };
        painter.rect_filled(rect, 6.0, node_fill);
        painter.rect_stroke(rect, 6.0, stroke, StrokeKind::Inside);
        painter.text(
            pos[v] - Vec2::new(0.0, 8.0),
            Align2::CENTER_CENTER,
            format!("v{v}"),
            title.clone(),
            text_color,
        );
        painter.text(
            pos[v] + Vec2::new(0.0, 9.0),
            Align2::CENTER_CENTER,
            label,
            small.clone(),
            text_color,
        );
    }
}

fn arrow(painter: &Painter, from: Pos2, to: Pos2, color: Color32) {
    painter.line_segment([from, to], Stroke::new(1.5, color));
    let dir = (to - from).normalized();
    let base = to - dir * ARROW_LEN;
    let side = Vec2::new(-dir.y, dir.x) * ARROW_HALF_WIDTH;
    painter.add(Shape::convex_polygon(
        vec![to, base + side, base - side],
        color,
        Stroke::NONE,
    ));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn circle_layout_is_distinct_and_inside_unit_disc() {
        let layout = Layout::circle(6, 4);
        assert_eq!(layout.places.len(), 6);
        assert_eq!(layout.transitions.len(), 4);
        let all: Vec<_> = layout
            .places
            .iter()
            .chain(&layout.transitions)
            .copied()
            .collect();
        for (i, a) in all.iter().enumerate() {
            assert!(a.to_vec2().length() <= 1.0 + 1e-5);
            for b in &all[i + 1..] {
                assert!(a.distance(*b) > 0.1);
            }
        }
    }

    #[test]
    fn single_transition_sits_in_the_middle() {
        let layout = Layout::circle(3, 1);
        assert_eq!(layout.transitions, vec![Pos2::ZERO]);
    }

    fn graph_of(
        build: fn() -> (Net, Marking),
    ) -> (ReachabilityGraph, Vec<usize>) {
        let (net, m0) = build();
        let graph = ReachabilityGraph::build(&net, m0, 100);
        let layer = layers(graph.markings.len(), &graph.edges);
        (graph, layer)
    }

    #[test]
    fn layers_are_shortest_distances_from_the_start() {
        use lab5::nets;
        // v0 -t1-> v1 -> {v2, v3} -> v4 -t4-> v0
        let (_, right) = graph_of(nets::right);
        assert_eq!(right, vec![0, 1, 2, 2, 3]);
        // v0 -> {v1, v2, v3} -> {v4, v5}; the edge v4 -> v0 goes back.
        let (_, left) = graph_of(nets::left);
        assert_eq!(left, vec![0, 1, 1, 1, 2, 2]);
    }

    #[test]
    fn layered_positions_form_columns_with_distinct_rows() {
        let positions = layered_positions(&[0, 1, 1, 1, 2, 2]);
        // x grows with the layer, vertices of a layer share x.
        assert_eq!(positions[0].x, -1.0);
        assert_eq!(positions[4].x, 1.0);
        assert!(
            positions[1].x == positions[2].x
                && positions[2].x == positions[3].x
        );
        // Spacing is common to all columns and the pictures stays inside.
        assert!((positions[2].y - positions[1].y - 1.0).abs() < 1e-5);
        assert!((positions[5].y - positions[4].y - 1.0).abs() < 1e-5);
        for p in &positions {
            assert!(p.x.abs() <= 1.0 && p.y.abs() <= 1.0);
        }
        // A single column sits in the middle.
        assert_eq!(layered_positions(&[0]), vec![Pos2::ZERO]);
    }

    #[test]
    fn bar_edge_leaves_through_nearest_side() {
        let c = Pos2::new(100.0, 100.0);
        let right = bar_edge(c, Pos2::new(300.0, 100.0));
        assert!((right.x - (100.0 + BAR_HALF.x)).abs() < 1e-4);
        let down = bar_edge(c, Pos2::new(100.0, 300.0));
        assert!((down.y - (100.0 + BAR_HALF.y)).abs() < 1e-4);
    }
}

use std::f32::consts::{FRAC_PI_2, TAU};

use common::colors;
use eframe::egui::{
    self, Align2, Color32, FontId, Painter, Pos2, Rect, Sense, Shape, Stroke,
    Vec2,
};
use lab5::net::{Marking, Net};

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
    let d = toward - center;
    if d == Vec2::ZERO {
        return center;
    }
    let tx = if d.x != 0.0 {
        BAR_HALF.x / d.x.abs()
    } else {
        f32::INFINITY
    };
    let ty = if d.y != 0.0 {
        BAR_HALF.y / d.y.abs()
    } else {
        f32::INFINITY
    };
    center + d * tx.min(ty)
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

    #[test]
    fn bar_edge_leaves_through_nearest_side() {
        let c = Pos2::new(100.0, 100.0);
        let right = bar_edge(c, Pos2::new(300.0, 100.0));
        assert!((right.x - (100.0 + BAR_HALF.x)).abs() < 1e-4);
        let down = bar_edge(c, Pos2::new(100.0, 300.0));
        assert!((down.y - (100.0 + BAR_HALF.y)).abs() < 1e-4);
    }
}

use crate::graph::ReachabilityGraph;
use crate::net::Net;

/// Structural classes; they overlap (a marked graph is also free-choice).
pub struct NetClass {
    /// Every transition has exactly one input and one output place.
    pub state_machine: bool,
    /// Every place has exactly one input and one output transition.
    pub marked_graph: bool,
    /// A place with several output transitions is the only input of each.
    pub free_choice: bool,
}

pub fn classify(net: &Net) -> NetClass {
    let state_machine = net
        .transitions
        .iter()
        .all(|t| t.source.len() == 1 && t.target.len() == 1);

    let producers = |p: usize| {
        net.transitions
            .iter()
            .filter(|t| t.target.contains(&p))
            .count()
    };
    let consumers = |p: usize| -> Vec<_> {
        net.transitions
            .iter()
            .filter(|t| t.source.contains(&p))
            .collect()
    };

    let marked_graph = (0..net.places.len())
        .all(|p| producers(p) == 1 && consumers(p).len() == 1);
    let free_choice = (0..net.places.len()).all(|p| {
        let out = consumers(p);
        out.len() <= 1 || out.iter().all(|t| t.source.len() == 1)
    });

    NetClass {
        state_machine,
        marked_graph,
        free_choice,
    }
}

/// Properties read off a reachability graph. When `complete` is false the
/// graph is only a part of the real one, so the values describe that part.
pub struct Properties {
    pub complete: bool,
    /// Largest number of tokens seen in each place.
    pub bounds: Vec<u32>,
    /// Vertices where no transition is enabled.
    pub deadlocks: Vec<usize>,
    /// The initial marking is reachable from every vertex.
    pub reversible: bool,
    /// From every vertex each transition can eventually fire.
    pub live: bool,
}

pub fn analyze(graph: &ReachabilityGraph, net: &Net) -> Properties {
    let vertices = graph.markings.len();

    let bounds = (0..net.places.len())
        .map(|p| graph.markings.iter().map(|m| m[p]).max().unwrap_or(0))
        .collect();

    let mut has_out = vec![false; vertices];
    let mut preds = vec![Vec::new(); vertices];
    for e in &graph.edges {
        has_out[e.from] = true;
        preds[e.to].push(e.from);
    }
    let deadlocks = (0..vertices).filter(|&v| !has_out[v]).collect();

    let reversible = reached_backwards(&preds, &[0]).iter().all(|&r| r);
    let live = (0..net.transitions.len()).all(|t| {
        let starts: Vec<_> = graph
            .edges
            .iter()
            .filter(|e| e.transition == t)
            .map(|e| e.from)
            .collect();
        reached_backwards(&preds, &starts).iter().all(|&r| r)
    });

    Properties {
        complete: graph.complete,
        bounds,
        deadlocks,
        reversible,
        live,
    }
}

/// Marks every vertex from which some vertex in `starts` can be reached.
fn reached_backwards(preds: &[Vec<usize>], starts: &[usize]) -> Vec<bool> {
    let mut reached = vec![false; preds.len()];
    let mut stack = Vec::new();
    for &s in starts {
        if !reached[s] {
            reached[s] = true;
            stack.push(s);
        }
    }
    while let Some(v) = stack.pop() {
        for &p in &preds[v] {
            if !reached[p] {
                reached[p] = true;
                stack.push(p);
            }
        }
    }
    reached
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nets;
    use crate::parse::parse;

    fn analyzed(text: &str, limit: usize) -> (NetClass, Properties) {
        let (net, m0) = parse(text).ok().unwrap();
        let graph = ReachabilityGraph::build(&net, m0, limit);
        (classify(&net), analyze(&graph, &net))
    }

    #[test]
    fn left_net() {
        let (class, p) = analyzed(nets::LEFT_TEXT, 10_000);
        assert!(!class.state_machine && !class.marked_graph);
        assert!(class.free_choice);
        assert!(p.complete);
        assert!(p.bounds.iter().all(|&b| b == 1));
        assert_eq!(p.deadlocks.len(), 1);
        assert!(!p.reversible && !p.live);
    }

    #[test]
    fn right_net() {
        let (class, p) = analyzed(nets::RIGHT_TEXT, 10_000);
        assert!(!class.state_machine);
        assert!(class.marked_graph && class.free_choice);
        assert!(p.complete);
        // Places in order of appearance: p1 p2 p3 p6 p4 p5.
        assert_eq!(p.bounds, vec![1, 1, 1, 2, 1, 1]);
        assert!(p.deadlocks.is_empty());
        assert!(p.reversible && p.live);
    }

    #[test]
    fn state_machine_class() {
        let (class, _) = analyzed("a: p1 -> p2\nb: p2 -> p1\nc: p2 -> p3", 100);
        assert!(class.state_machine && class.free_choice);
        assert!(!class.marked_graph);
    }

    #[test]
    fn conflict_with_extra_input_is_not_free_choice() {
        // p1 feeds both transitions, but b also needs p2.
        let (class, _) =
            analyzed("a: p1 -> p3\nb: p1 p2 -> p3\nmarking: p1=1 p2=1", 100);
        assert!(!class.free_choice);
    }

    #[test]
    fn no_deadlock_but_not_reversible_or_live() {
        // After t0 the token circles between p2 and p3 and never returns.
        let (_, p) = analyzed(
            "t0: p1 -> p2\nt1: p2 -> p3\nt2: p3 -> p2\nmarking: p1=1",
            100,
        );
        assert!(p.deadlocks.is_empty());
        assert!(!p.reversible && !p.live);
    }

    #[test]
    fn unbounded_net_is_flagged_incomplete() {
        let (_, p) = analyzed("t: -> p1", 20);
        assert!(!p.complete);
    }
}

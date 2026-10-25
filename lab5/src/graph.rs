use std::collections::{HashMap, VecDeque};

use crate::net::{Marking, Net};

pub struct Edge {
    pub from: usize,
    pub transition: usize,
    pub to: usize,
}

pub struct ReachabilityGraph {
    /// Vertex `i` is `markings[i]`; vertex 0 is the initial marking.
    pub markings: Vec<Marking>,
    pub edges: Vec<Edge>,
    /// False when the search stopped at the vertex limit, so the graph may
    /// miss reachable markings.
    pub complete: bool,
    index: HashMap<Marking, usize>,
}

impl ReachabilityGraph {
    pub fn index_of(&self, marking: &Marking) -> Option<usize> {
        self.index.get(marking).copied()
    }

    /// All `(vertex, t, u)` with `t < u` such that `t` and `u` can fire in
    /// parallel in that vertex's marking.
    pub fn concurrent_pairs(&self, net: &Net) -> Vec<(usize, usize, usize)> {
        let count = net.transitions.len();
        let mut pairs = Vec::new();
        for (vertex, marking) in self.markings.iter().enumerate() {
            for t in 0..count {
                for u in t + 1..count {
                    if net.can_fire_concurrently(marking, t, u) {
                        pairs.push((vertex, t, u));
                    }
                }
            }
        }
        pairs
    }

    /// Breadth-first search over markings. Every enabled firing becomes an
    /// edge, including those leading to an already known marking.
    pub fn build(net: &Net, initial: Marking, limit: usize) -> Self {
        let mut graph = Self {
            markings: vec![initial.clone()],
            edges: Vec::new(),
            complete: true,
            index: HashMap::from([(initial, 0)]),
        };
        let mut queue = VecDeque::from([0]);

        while let Some(from) = queue.pop_front() {
            let marking = graph.markings[from].clone();
            for transition in net.enabled_transitions(&marking) {
                let next = net.fire(&marking, transition).unwrap();
                let to = match graph.index.get(&next) {
                    Some(&to) => to,
                    None if graph.markings.len() >= limit => {
                        graph.complete = false;
                        continue;
                    }
                    None => {
                        let to = graph.markings.len();
                        graph.markings.push(next.clone());
                        graph.index.insert(next, to);
                        queue.push_back(to);
                        to
                    }
                };
                graph.edges.push(Edge {
                    from,
                    transition,
                    to,
                });
            }
        }
        graph
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::net::Transition;
    use crate::nets;

    #[test]
    fn right_net_graph() {
        let (net, m0) = nets::right();
        let g = ReachabilityGraph::build(&net, m0.clone(), 10_000);
        assert!(g.complete);
        assert_eq!(g.markings.len(), 5);
        assert_eq!(g.edges.len(), 6);
        assert_eq!(g.index_of(&m0), Some(0));
        assert!(g.index_of(&vec![0, 0, 0, 1, 1, 2]).is_some());
        assert!(g.index_of(&vec![1, 1, 0, 0, 0, 1]).is_none());
    }

    #[test]
    fn left_net_graph() {
        let (net, m0) = nets::left();
        let g = ReachabilityGraph::build(&net, m0, 10_000);
        assert!(g.complete);
        assert_eq!(g.markings.len(), 6);
        assert_eq!(g.edges.len(), 8);
        // (p2, p5) is a dead end: nothing is enabled there.
        let dead = g.index_of(&vec![0, 1, 0, 0, 1]).unwrap();
        assert!(g.edges.iter().all(|e| e.from != dead));
    }

    #[test]
    fn right_net_concurrency_only_after_fork() {
        let (net, m0) = nets::right();
        let g = ReachabilityGraph::build(&net, m0, 10_000);
        let b = g.index_of(&vec![0, 1, 1, 0, 0, 2]).unwrap();
        // t2 and t3 only.
        assert_eq!(g.concurrent_pairs(&net), vec![(b, 1, 2)]);
    }

    #[test]
    fn left_net_concurrency_skips_conflict() {
        let (net, m0) = nets::left();
        let g = ReachabilityGraph::build(&net, m0, 10_000);
        // a, b, d enabled in the initial marking: a-b and b-d are parallel,
        // a-d compete for the single token in p4.
        assert_eq!(g.concurrent_pairs(&net), vec![(0, 0, 1), (0, 1, 3)]);
    }

    #[test]
    fn shared_place_with_enough_tokens_is_not_a_conflict() {
        let (net, _) = nets::left();
        // Two tokens in p4: both a and d can take one.
        assert!(net.can_fire_concurrently(&vec![0, 0, 0, 2, 0], 0, 3));
        assert!(!net.can_fire_concurrently(&vec![0, 0, 0, 1, 0], 0, 3));
    }

    #[test]
    fn unbounded_net_stops_at_limit() {
        // One transition that returns its token and adds another.
        let net = Net {
            places: vec!["p1".to_string()],
            transitions: vec![Transition {
                name: "t".to_string(),
                source: vec![0],
                target: vec![0, 0],
            }],
        };
        let g = ReachabilityGraph::build(&net, vec![1], 50);
        assert!(!g.complete);
        assert_eq!(g.markings.len(), 50);
    }

    #[test]
    fn limit_equal_to_vertex_count_is_complete() {
        let (net, m0) = nets::right();
        let g = ReachabilityGraph::build(&net, m0.clone(), 5);
        assert!(g.complete);
        let g = ReachabilityGraph::build(&net, m0, 4);
        assert!(!g.complete);
    }
}

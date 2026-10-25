use crate::net::{Marking, Net, Transition};

/// The same nets in the text format of `parse::parse`.
pub const LEFT_TEXT: &str = "\
a: p4 -> p1
b: p3 -> p2
c: p1 p2 -> p3 p4
d: p4 -> p5
marking: p3=1 p4=1
";

pub const RIGHT_TEXT: &str = "\
t1: p1 -> p2 p3 p6
t2: p2 -> p4
t3: p3 -> p5
t4: p4 p5 p6 -> p1
marking: p1=1 p6=1
";

fn transition(name: &str, source: &[usize], target: &[usize]) -> Transition {
    Transition {
        name: name.to_string(),
        source: source.to_vec(),
        target: target.to_vec(),
    }
}

fn places(count: usize) -> Vec<String> {
    (1..=count).map(|i| format!("p{i}")).collect()
}

/// Left net of figure 1: transitions a, b, c, d over places p1..p5.
pub fn left() -> (Net, Marking) {
    let net = Net {
        places: places(5),
        transitions: vec![
            transition("a", &[3], &[0]),
            transition("b", &[2], &[1]),
            transition("c", &[0, 1], &[2, 3]),
            transition("d", &[3], &[4]),
        ],
    };
    (net, vec![0, 0, 1, 1, 0])
}

/// Right net of figure 1: transitions t1..t4 over places p1..p6.
pub fn right() -> (Net, Marking) {
    let net = Net {
        places: places(6),
        transitions: vec![
            transition("t1", &[0], &[1, 2, 5]),
            transition("t2", &[1], &[3]),
            transition("t3", &[2], &[4]),
            transition("t4", &[3, 4, 5], &[0]),
        ],
    };
    (net, vec![1, 0, 0, 0, 0, 1])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn right_net_fork_and_join() {
        let (net, m0) = right();
        assert_eq!(net.enabled_transitions(&m0), vec![0]);

        let m1 = net.fire(&m0, 0).unwrap();
        assert_eq!(m1, vec![0, 1, 1, 0, 0, 2]);
        assert_eq!(net.enabled_transitions(&m1), vec![1, 2]);

        // t4 needs both branches to finish.
        let m2 = net.fire(&m1, 1).unwrap();
        assert!(!net.is_enabled(&m2, 3));
        let m3 = net.fire(&m2, 2).unwrap();
        assert!(net.is_enabled(&m3, 3));

        assert_eq!(net.fire(&m3, 3).unwrap(), m0);
    }

    #[test]
    fn fire_disabled_returns_none() {
        let (net, m0) = right();
        assert!(net.fire(&m0, 3).is_none());
    }

    #[test]
    fn fire_does_not_mutate_input() {
        let (net, m0) = right();
        let copy = m0.clone();
        net.fire(&m0, 0);
        assert_eq!(m0, copy);
    }

    #[test]
    fn left_net_conflict_and_cycle() {
        let (net, m0) = left();
        // p3 feeds b, p4 feeds both a and d (conflict); c lacks p1, p2.
        assert_eq!(net.enabled_transitions(&m0), vec![0, 1, 3]);

        let m1 = net.fire(&m0, 1).unwrap();
        assert_eq!(m1, vec![0, 1, 0, 1, 0]);
        let m2 = net.fire(&m1, 0).unwrap();
        assert_eq!(m2, vec![1, 1, 0, 0, 0]);
        assert_eq!(net.enabled_transitions(&m2), vec![2]);
        assert_eq!(net.fire(&m2, 2).unwrap(), m0);
    }

    #[test]
    fn left_net_d_loses_token() {
        let (net, m0) = left();
        let m1 = net.fire(&m0, 3).unwrap();
        assert_eq!(m1, vec![0, 0, 1, 0, 1]);
        // p5 has no outgoing transition, the token stays there.
        assert_eq!(net.enabled_transitions(&m1), vec![1]);
    }
}

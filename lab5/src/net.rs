/// Number of tokens in each place, indexed like `Net::places`.
pub type Marking = Vec<u32>;

pub struct Transition {
    pub name: String,
    /// Input places: one token is taken from each.
    pub source: Vec<usize>,
    /// Output places: one token is put into each.
    pub target: Vec<usize>,
}

pub struct Net {
    pub places: Vec<String>,
    pub transitions: Vec<Transition>,
}

impl Net {
    /// A transition is enabled when every input place holds a token.
    pub fn is_enabled(&self, marking: &Marking, transition: usize) -> bool {
        self.transitions[transition]
            .source
            .iter()
            .all(|&p| marking[p] > 0)
    }

    pub fn enabled_transitions(&self, marking: &Marking) -> Vec<usize> {
        (0..self.transitions.len())
            .filter(|&t| self.is_enabled(marking, t))
            .collect()
    }

    /// Two transitions can fire in parallel when every place holds enough
    /// tokens for both of them together.
    pub fn can_fire_concurrently(
        &self,
        marking: &Marking,
        t: usize,
        u: usize,
    ) -> bool {
        let mut need = vec![0; marking.len()];
        for &p in self.transitions[t]
            .source
            .iter()
            .chain(&self.transitions[u].source)
        {
            need[p] += 1;
        }
        need.iter().zip(marking).all(|(n, m)| n <= m)
    }

    /// Returns the marking after firing, or `None` if the transition is not
    /// enabled. The input marking is left untouched.
    pub fn fire(
        &self,
        marking: &Marking,
        transition: usize,
    ) -> Option<Marking> {
        if !self.is_enabled(marking, transition) {
            return None;
        }
        let t = &self.transitions[transition];
        let mut next = marking.clone();
        for &p in &t.source {
            next[p] -= 1;
        }
        for &p in &t.target {
            next[p] += 1;
        }
        Some(next)
    }
}

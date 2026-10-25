pub enum Kind {
    Real,
    Count { min: u32 },
    Flag,
}

pub struct Param {
    pub name: &'static str,
    pub value: f64,
    pub kind: Kind,
}

impl Param {
    pub fn real(name: &'static str, value: f64) -> Self {
        Self {
            name,
            value,
            kind: Kind::Real,
        }
    }

    pub fn count(name: &'static str, value: u32, min: u32) -> Self {
        Self {
            name,
            value: value as f64,
            kind: Kind::Count { min },
        }
    }

    pub fn flag(name: &'static str, value: bool) -> Self {
        Self {
            name,
            value: if value { 1.0 } else { 0.0 },
            kind: Kind::Flag,
        }
    }
}

pub struct Metric {
    pub name: &'static str,
    pub value: f64,
}

pub struct Solution {
    pub labels: Vec<String>,
    pub p: Vec<f64>,
    pub metrics: Vec<Metric>,
    pub warnings: Vec<String>,
}

impl Solution {
    pub fn new(labels: Vec<String>, p: Vec<f64>) -> Self {
        Self {
            labels,
            p,
            metrics: Vec::new(),
            warnings: Vec::new(),
        }
    }

    pub fn metric(mut self, name: &'static str, value: f64) -> Self {
        self.metrics.push(Metric { name, value });
        self
    }

    pub fn warning(mut self, text: impl Into<String>) -> Self {
        self.warnings.push(text.into());
        self
    }

    #[cfg(test)]
    pub fn find(&self, name: &str) -> Option<f64> {
        self.metrics
            .iter()
            .find(|metric| metric.name == name)
            .map(|metric| metric.value)
    }
}

pub struct Task {
    pub name: &'static str,
    pub params: Vec<Param>,
    pub solve: fn(&[f64]) -> Result<Solution, String>,
}

pub fn require_positive(name: &str, value: f64) -> Result<(), String> {
    if value > 0.0 && value.is_finite() {
        Ok(())
    } else {
        Err(format!("{name} must be positive"))
    }
}

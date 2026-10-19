/// The distribution under study. Everything the rest of the program knows
/// about it lives here: to switch distributions, rewrite this file.
pub struct Params {}

impl Params {
    fn lambda() -> f64 {
        2.0
    }

    pub fn f(x: f64) -> f64 {
        if x < 0.0 {
            0.0
        } else {
            Self::lambda() * (-Self::lambda() * x).exp()
        }
    }

    pub fn g(x: f64) -> f64 {
        if x < 0.0 {
            0.0
        } else {
            1.0 - (-Self::lambda() * x).exp()
        }
    }

    pub fn g_inv(x: f64) -> f64 {
        -(1.0 - x).ln() / Self::lambda()
    }

    pub fn mean() -> f64 {
        1.0 / Self::lambda()
    }

    pub fn variance() -> f64 {
        1.0 / (Self::lambda() * Self::lambda())
    }

    pub fn range() -> (f64, f64) {
        (0.0, 3.0)
    }
}

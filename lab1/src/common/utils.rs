pub fn parse_in_range(
    s: &str,
    min: f64,
    max: f64,
) -> Result<f64, Box<dyn std::error::Error + Send + Sync>> {
    let v = s.parse::<f64>()?;
    if !(min..=max).contains(&v) {
        return Err(format!("value {v} out of range [{min}, {max}]").into());
    }
    Ok(v)
}

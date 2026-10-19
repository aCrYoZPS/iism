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

pub fn parse_probability_list(input: &str) -> Result<Vec<f64>, String> {
    input
        .split(|c: char| c == ',' || c.is_whitespace())
        .filter(|part| !part.is_empty())
        .map(|part| {
            parse_in_range(part, 0.0, 1.0).map_err(|error| error.to_string())
        })
        .collect()
}

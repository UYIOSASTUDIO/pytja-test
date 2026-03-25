#[allow(dead_code)]
pub fn format_bytes(b: u64) -> String {
    const UNIT: u64 = 1024;
    if b < UNIT { return format!("{} B", b); }
    let div = UNIT as f64;
    let exp = (b as f64).ln() / div.ln();
    let pre = "KMGTPE".chars().nth(exp as usize - 1).unwrap_or('?');
    format!("{:.1} {}B", (b as f64) / div.powi(exp as i32), pre)
}
pub(crate) fn string_number(text: &str) -> f64 {
    let text = text.trim_matches(|c| matches!(c, '\u{0009}'..='\u{000d}' | ' ' | '\u{00a0}' | '\u{1680}' | '\u{2000}'..='\u{200a}' | '\u{2028}' | '\u{2029}' | '\u{202f}' | '\u{205f}' | '\u{3000}' | '\u{feff}'));
    if text.is_empty() {
        return 0.0;
    }
    match text {
        "Infinity" | "+Infinity" => return f64::INFINITY,
        "-Infinity" => return f64::NEG_INFINITY,
        _ => {}
    }
    if text.starts_with('0') && text.len() >= 2 {
        let radix = match text.as_bytes()[1] {
            b'x' | b'X' => Some(16),
            b'o' | b'O' => Some(8),
            b'b' | b'B' => Some(2),
            _ => None,
        };
        if let Some(radix) = radix {
            let digits = &text.as_bytes()[2..];
            if digits.is_empty() || !digits.iter().all(|c| (*c as char).is_digit(radix)) {
                return f64::NAN;
            }
            return num_bigint::BigUint::parse_bytes(digits, radix)
                .and_then(|value| value.to_str_radix(10).parse().ok())
                .unwrap_or(f64::NAN);
        }
    }
    if !text
        .bytes()
        .all(|c| matches!(c, b'0'..=b'9' | b'.' | b'+' | b'-' | b'e' | b'E'))
    {
        return f64::NAN;
    }
    text.parse().unwrap_or(f64::NAN)
}

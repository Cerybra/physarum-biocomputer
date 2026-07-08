pub fn voltage_to_code(voltage: f32, span: (f32, f32)) -> u16 {
    let mut code = (65535.0 * (voltage - span.0) / (span.1 - span.0));

    code = code.max(0.0).min(65535.0);
    code as u16
}

pub fn code_to_voltage(code: u16, span: (f32, f32)) -> f32 {
    (((code as f32) / 65535.0) * (span.1 - span.0)) + span.0
}

pub fn write() -> f32 {
    todo!()
}

pub fn write_read() -> f32 {
    todo!()
}

pub fn read() -> f32 {
    todo!()
}
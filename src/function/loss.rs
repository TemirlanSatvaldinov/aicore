pub fn cross_entropy (target:&[f32],predict:&[f32]) ->f32 {
    let input_len = target.len();
    let mut loss = 0.0f32;
    for i in 0..input_len {
        loss += f32::ln(predict[i]) * target[i];
    }
    return  -(loss);
}
pub fn mse (target:&[f32],predict:&[f32]) ->f32 {
    let input_len = target.len();
    let mut loss = 0.0f32;
    for i in 0..input_len {
        loss += f32::powf(target[i] + predict[i], 2.0)
    }
    return  loss / (input_len as f32);
}
use std::f32::INFINITY;



const  MIN_RELU_VAL:f32 = 0.00;

pub fn relu (x:f32) -> f32 {
    if x <=0.0 {
        return MIN_RELU_VAL  ;
    }
    return  x;
}
pub fn relu_derivative(x:f32) -> f32 {
    
    if x <= 0.0 {
        return MIN_RELU_VAL;
    }
    return 1.0f32;
}
pub fn sigmoid (x:f32) ->f32 {
    return 1.00 / (1.00 + f32::exp(-x));
}
pub fn sigmoid_derivative (x:f32) ->f32 {
    return x * (1.00 -x)
}
pub fn  softmax(input:&mut[f32]) {
    let mut m = -INFINITY;
    let input_len = input.len();
    for i in 0..input_len {
        if input[i] > m {
            m = input[i]
        }
    }
    let mut sum:f32= 0.0f32;
    for i in 0..input_len {
        sum += f32::exp(input[i] - m);      
    }
    let offset = m + f32::ln(sum);
    for i in 0..input_len {
        input[i] = f32::exp(input[i] - offset);
    }

}
// pub fn  softmax_derivative(input:&Vec<f32>,dst: &mut Vec<f32>) {
//     let input_len = input.len();
//     for i in 0..input_len{
//         dst[i]  +=  1.0f32 * input[i]
//     }
    
// }


pub fn sgd_update ( weight:&mut [f32],dl_w:&[f32],lr_rate:f32, l2:f32,train_size:f32) {
       
    let len = weight.len();
    if l2 > 0.0f32 {
        for i in 0..len {
            weight[i] = (weight[i] * (1.0f32 - (lr_rate * l2 / train_size))) + lr_rate * dl_w[i];
        }
        return;
    }
    for i in 0..len {   
        weight[i] += lr_rate * dl_w[i] ;
    }

}
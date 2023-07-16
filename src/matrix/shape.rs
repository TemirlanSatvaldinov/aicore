


// pub fn print_matrix (input:&Vec<f32>,row:usize,col:usize) {
//     for i in 0..row {
//         for j in 0..col {
//             let x =i * col ;
//             print!("{:.3} ",input[x +j])
//         }
//         println!("")
//     }
//     println!("")
// }


pub fn transpose_matrix (input:&[f32],input_h:usize,input_w:usize,dst:&mut [f32]) {
    let dst_h = input_w;
    let dst_w = input_h;
    
    for  row in 0..dst_h {
        let x = row * dst_w;
        for col in 0..dst_w {
            dst [x + col] = input[col * input_w +row]
        }
    }
}
pub fn rotate_matrix180 (input:&[f32],input_h:usize,input_w:usize,dst:&mut [f32]) {
    let mut i = 0;
    for  row in (0..input_h).rev() {
        for col in (0..input_w).rev(){
            dst[i]=input[row * input_w + col];
            i+=1;
        }
    }
    
}

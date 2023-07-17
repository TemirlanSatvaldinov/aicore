pub fn matrix_mul(a:& [f32],row_a:usize,_col_a:usize,b:&[f32],row_b:usize,col_b:usize,c:&mut [f32]) {
    
    //assert_eq!(col_a,row_b,"trying multiple {}x{} with {}x{}", row_a,col_a,row_b,col_b);
    
    for i in 0..row_a {
        let  slice_c = &mut c[i*col_b..];
        for j in 0..col_b {
            slice_c[j] = 0.0;
        }
        for k in 0..row_b {
            let slice_b = &b[k*col_b..];
            let x = a[i*row_b+k];
            for j in 0..col_b {
                slice_c[j] += x *slice_b[j]
            }
        }
    }
}
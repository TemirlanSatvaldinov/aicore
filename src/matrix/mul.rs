pub fn matrix_mul(A:& [f32],row_a:usize,col_a:usize,B:&[f32],row_b:usize,col_b:usize,C:&mut [f32]) {
    
    //assert_eq!(col_a,row_b,"trying multiple {}x{} with {}x{}", row_a,col_a,row_b,col_b);
    
    for i in 0..row_a {
        let mut c = &mut C[i*col_b..];
        for j in 0..col_b {
            c[j] = 0.0;
        }
        for k in 0..row_b {
            let b = &B[k*col_b..];
            let a = A[i*row_b+k];
            for j in 0..col_b {
                c[j] += a *b[j]
            }
        }
    }
}
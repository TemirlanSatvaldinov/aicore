use super::super::simd::gemm_kernel::kernel8x8;
use super::super::simd::pack::{pack_matrix_a,pack_matrix_b};

pub fn matrix_mul(a:& [f32],row_a:usize,col_a:usize,b:&[f32],row_b:usize,col_b:usize,c:&mut [f32]) {
    
    assert_eq!(col_a,row_b,"trying multiple {}x{} with {}x{}", row_a,col_a,row_b,col_b);

    #[cfg(any( target_arch="x86_64"))]
    {
        if is_x86_feature_detected!("fma") && is_x86_feature_detected!("avx") {
           
            return sgemm_fma_avx (row_a,col_a,col_b,col_a,col_b,col_b,a,b,c) 
            
        }
    }        
    matrix_mul_fallback(row_a,col_a,col_b,a,b,c)
    
}

pub fn matrix_mul_fallback(m:usize,k:usize,n:usize,a:& [f32],b:&[f32],c:&mut [f32]) {

    for i in 0..m {
        let  c_slice = &mut c[i*n..];
        for j in 0..n {
            c_slice[j] = 0.0;
        }
        for p in 0..k {
            let b_slice = &b[p*n..];
            let x = a[i*k+p];
            for j in 0..n {
                c_slice[j] += x *b_slice[j]
            }
        }
    }

 }

//
fn sgemm_fma_avx(m:usize,k:usize,n:usize ,lda:usize,ldb:usize,ldc:usize,a:&[f32],b:&[f32],c:&mut [f32]) {
    let (m8,n8,k8) = align_size(m,n,k,8,8);
    
    let kc = usize::min(256,k8) ;
    let mc = usize::min(128,m8) ;
    let nc = usize::min(4 * 1024 * 1024 /4/kc,n8);
       
    for j in (0..n).step_by(nc) {
        let n_block = usize::min(n - j,nc);
        
        for  p in (0..k).step_by(kc) {
            let k_block = usize::min(k - p,  kc); 
            let mut buf_b = vec![0.0f32;kc * nc]; 

            for i in (0..m).step_by(mc) {
                let m_block = usize::min(m - i , mc);  
                
                unsafe {
                    let mut buf_a = vec![0.0f32;kc * mc];
                   
                    let slice_b = &b[p*ldb + j..];
                    let slice_a = &a[i*lda + p..];
                    let slice_c = &mut c[i*ldc + j..];

                    pack_matrix_a(m_block, k_block, slice_a, lda,  &mut buf_a);
                    for  jj in (0..n_block).step_by(8) {
                        if i==0 {
                            pack_matrix_b(k_block, &slice_b[jj..], ldb,  &mut buf_b[k_block*jj..],8,n_block-jj);   
                        }
                        for ii in (0..m_block).step_by(8) { 
                            kernel8x8(k_block, &buf_a[ii*k_block..], 1 ,8, &buf_b[k_block*jj..], 8, &mut slice_c[ii*ldc+jj..], ldc ,m_block-ii)    
                        }
                    } 
                }
            }
        }
    }
   
}
fn align_size (m:usize,n:usize,k:usize,kh:usize,kw:usize) -> (usize,usize,usize) {
    let (mut m8,mut k8,mut n8) = (m,k,n);

    if m%8 != 0 {
        m8 = m / kh * kh + kh;
    }
    if k%8 != 0 {
        k8 = k / kh * kh + kh;
    }
    if n%8 != 0 {
        n8 = n / kw * kw + kw;
    }
    return (m8,n8,k8);
}

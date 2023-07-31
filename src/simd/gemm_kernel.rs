use std::arch::x86_64::{*};

#[target_feature(enable = "avx,fma")]
#[cfg(any( target_arch = "x86_64"))]
pub unsafe fn kernel8x8 (k:usize,a:&[f32],lda:usize,stride:usize,b:&[f32],ldb:usize,c:&mut [f32],ldc:usize,edge:usize) {

    let mut c_row = [_mm256_setzero_ps();8]; 

    let mut ptr_b = b.as_ptr();
    let mut ptr_a = a.as_ptr();
    let mut ptr_c = c.as_mut_ptr();
    
    for _ in 0..k {
        let b0 = _mm256_loadu_ps(ptr_b);

        let mut a0 = _mm256_set1_ps(*ptr_a);
        c_row[0] =_mm256_fmadd_ps(a0, b0, c_row[0]);

        a0 = _mm256_set1_ps(*ptr_a.add(lda));
        c_row[1] = _mm256_fmadd_ps(a0, b0, c_row[1]);

        a0 = _mm256_set1_ps(*ptr_a.add(2 * lda));
        c_row[2] =_mm256_fmadd_ps(a0, b0, c_row[2]);

        a0 = _mm256_set1_ps(*ptr_a.add(3 * lda ));
        c_row[3] =_mm256_fmadd_ps(a0, b0, c_row[3]);

        a0 = _mm256_set1_ps(*ptr_a.add(4 * lda));
        c_row[4] =_mm256_fmadd_ps(a0, b0, c_row[4]);

        a0 = _mm256_set1_ps(*ptr_a.add(5 * lda));
        c_row[5] =_mm256_fmadd_ps(a0, b0, c_row[5]);

        a0 = _mm256_set1_ps(*ptr_a.add(6 * lda ));
        c_row[6] =_mm256_fmadd_ps(a0, b0, c_row[6]);

        a0 = _mm256_set1_ps(*ptr_a.add(7 * lda ));
        c_row[7] =_mm256_fmadd_ps(a0, b0, c_row[7]);
        
        ptr_b = ptr_b.add(ldb); ptr_a = ptr_a.add(stride)                 
    } 
    if edge  < 8 {
        for i in 0..edge {
            _mm256_storeu_ps(ptr_c , _mm256_add_ps(c_row[i], _mm256_loadu_ps(ptr_c)));
            ptr_c = ptr_c.add(ldc );
        }
        return;
    }
   
    _mm256_storeu_ps(ptr_c , _mm256_add_ps(c_row[0], _mm256_loadu_ps(ptr_c)));
    
    ptr_c = ptr_c.add(ldc );
    _mm256_storeu_ps(ptr_c , _mm256_add_ps(c_row[1], _mm256_loadu_ps(ptr_c)));

    ptr_c = ptr_c.add(ldc );
    _mm256_storeu_ps(ptr_c , _mm256_add_ps(c_row[2], _mm256_loadu_ps(ptr_c)));
    
    ptr_c = ptr_c.add(ldc );
    _mm256_storeu_ps(ptr_c , _mm256_add_ps(c_row[3], _mm256_loadu_ps(ptr_c)));
    
    ptr_c = ptr_c.add(ldc );
    _mm256_storeu_ps(ptr_c , _mm256_add_ps(c_row[4], _mm256_loadu_ps(ptr_c)));

    ptr_c = ptr_c.add(ldc );
    _mm256_storeu_ps(ptr_c , _mm256_add_ps(c_row[5], _mm256_loadu_ps(ptr_c)));
    
    ptr_c = ptr_c.add(ldc );
    _mm256_storeu_ps(ptr_c , _mm256_add_ps(c_row[6], _mm256_loadu_ps(ptr_c)));

    ptr_c = ptr_c.add(ldc );
    _mm256_storeu_ps(ptr_c , _mm256_add_ps(c_row[7], _mm256_loadu_ps(ptr_c)));
    
}
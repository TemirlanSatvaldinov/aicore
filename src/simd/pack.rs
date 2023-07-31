use std::arch::x86_64::{*};


#[target_feature(enable = "avx,fma")]
#[cfg(any( target_arch = "x86_64"))]
pub unsafe fn pack_matrix_b (k:usize,b:&[f32],ldb:usize,dst:&mut  [f32],kw:usize,edge:usize) {
    let mut ptr_dst = dst.as_mut_ptr();
    let mut ptr_b = b.as_ptr();
  
    if edge >= kw {
            for _ in 0..k {
                _mm256_storeu_ps(ptr_dst, _mm256_loadu_ps(ptr_b)); 
                ptr_dst = ptr_dst.add(8); ptr_b = ptr_b.add(ldb) 
                
            }
        return;
    }
    // src
    // 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,16.0, 17.0, 40.0, 41.0, 
    // 20.0, 21.0, 22.0, 23.0, 24.0, 25.0,26.0, 27.0, 50.0, 51.0, 
    // 30.0, 31.0, 32.0, 33.0, 34.0, 35.0,36.0, 37.0, 60.0, 61.0;
    // dst
    // 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,16.0, 17.0, 
    // 20.0, 21.0, 22.0, 23.0, 24.0, 25.0,26.0, 27.0,  
    // 30.0, 31.0, 32.0, 33.0, 34.0, 35.0,36.0, 37.0, 
    // 40.0, 41.0, 0.0,  0.0,  0.0,  0.0, 0.0,  0.0,
    // 50.0, 51.0, 0.0,  0.0,  0.0,  0.0, 0.0,  0.0,
    // 60.0, 61.0  0.0,  0.0,  0.0,  0.0, 0.0,  0.0,

    
        
    if edge < 8 {
        for row in 0..k {
            for col in 0..edge {
                *ptr_dst.add( row * kw + col) = *ptr_b.add(row * ldb + col);
            }
        }       
    }
    
}


#[target_feature(enable = "avx,fma")]
#[cfg(any( target_arch = "x86_64"))]
pub unsafe fn pack_matrix_a (m:usize,k:usize,a:&[f32],lda:usize,dst:&mut [f32]) {
     
    let mut ptr_a = a.as_ptr();
    let mut ptr_dst = dst.as_mut_ptr();
    
    let strs = 8;let width = 4;
    let mut row_counter = 0;
    
    if m  < 8 {
        for  row in 0..m {
            for col in 0..k {
                *ptr_dst.add( col * strs + row) = *ptr_a.add(row * lda + col);
            }
        }
        return;
    }
    
    for _ in (0..m-7).step_by(strs) {
        let mut col_counter = 0;
        if k >= 4 {
            for k in (0..k-3).step_by(width) {
            
                let  p_a = ptr_a.add(k);

                let  a0 = _mm_loadu_ps(p_a);                     //   10 11 12 13
                let  a1 = _mm_loadu_ps(p_a.add(lda )); //   20 21 22 23
                let  a2 = _mm_loadu_ps(p_a.add(2 * lda )); //   30 31 32 33
                let  a3 = _mm_loadu_ps(p_a.add(3 * lda )); //   40 41 42 43
                let  a4 = _mm_loadu_ps(p_a.add(4 * lda )); //   50 51 52 53
                let  a5 = _mm_loadu_ps(p_a.add(5 * lda )); //   60 61 62 63
                let  a6 = _mm_loadu_ps(p_a.add(6 * lda )); //   70 71 72 73
                let  a7 = _mm_loadu_ps(p_a.add(7 * lda )); //   80 81 82 83
        
                
                let a0a2o = _mm_unpacklo_ps(a0, a2);
                let a1a3o = _mm_unpacklo_ps(a1, a3);
                let dst00 = _mm_unpacklo_ps(a0a2o, a1a3o);
                
                let a4a6o = _mm_unpacklo_ps(a4,a6);
                let a5a7o = _mm_unpacklo_ps(a5,a7);
                let dst01 = _mm_unpacklo_ps(a4a6o,a5a7o);
                let dst11 = _mm_unpackhi_ps(a4a6o,a5a7o);
    
                let a0a2i = _mm_unpackhi_ps(a0, a2);
                let a1a3i = _mm_unpackhi_ps(a1, a3);
                let dst20 = _mm_unpacklo_ps(a0a2i, a1a3i);
                let dst30 = _mm_unpackhi_ps(a0a2i, a1a3i);
    
    
                let a4a6i = _mm_unpackhi_ps(a4,a6);
                let a5a7i = _mm_unpackhi_ps(a5,a7);
                let dst31 = _mm_unpackhi_ps(a4a6i,a5a7i);
                // 10 20 11 21 -> (10 11 12 13), (20 21 22 23)
                let a0a1o = _mm_unpacklo_ps(a0,a1);
                // 30 40 31 41
                let a2a3o = _mm_unpacklo_ps(a2,a3);
                // 11  21 31 41  -> (30 40 31 41),(10 20 11 21) 
                let dst10 = _mm_movehl_ps(a2a3o,a0a1o);
                let a4a5o = _mm_unpackhi_ps(a4,a5);
                let a6a7o = _mm_unpackhi_ps(a6,a7);
                let dst21 = _mm_movelh_ps(a4a5o, a6a7o);
                
                /*
                10 20 30 40 50 60 70 80
                11 21 31 41 51 61 71 81
                12 22 32 42 52 62 72 82
                13 23 33 43 53 63 72 83
                */
                _mm_storeu_ps(ptr_dst,dst00);
                _mm_storeu_ps(ptr_dst.add(4),dst01);
                _mm_storeu_ps(ptr_dst.add(8),dst10);
                _mm_storeu_ps(ptr_dst.add(12),dst11);
                _mm_storeu_ps(ptr_dst.add(16),dst20);
                _mm_storeu_ps(ptr_dst.add(20),dst21);
                _mm_storeu_ps(ptr_dst.add(24),dst30);
                _mm_storeu_ps(ptr_dst.add(28),dst31);
                
                col_counter +=  width ;
                ptr_dst = ptr_dst.add( strs * width);
                
            }
        }
        // width
        if col_counter < k  {
            for k in col_counter..k  {
               
                let  p_a = ptr_a.add(k);

                let  a0 = *p_a;                     
                let  a1 = *p_a.add(1 * lda ); 
                let  a2 = *p_a.add(2 * lda ); 
                let  a3 = *p_a.add(3 * lda ); 
                let  a4 = *p_a.add(4 * lda  ); 
                let  a5 = *p_a.add(5 * lda  ); 
                let  a6 = *p_a.add(6 * lda  ); 
                let  a7 = *p_a.add(7 * lda );
                
                *ptr_dst = a0;
                *ptr_dst.add(1) = a1;
                *ptr_dst.add(2) = a2;
                *ptr_dst.add(3) = a3;
                *ptr_dst.add(4) = a4;
                *ptr_dst.add(5) = a5;
                *ptr_dst.add(6) = a6;
                *ptr_dst.add(7) = a7;
                ptr_dst = ptr_dst.add(8);
                               
            }
           
                /*
                    10 11 12     10 20 .. 80
                    20 21 22  -> 11 21 .. 81
                     . .  .      12 22 .. 82     
                    80 81 82     0  0 .. 0
                */
            //ptr_dst = ptr_dst.add((edge * 8));
        }
        row_counter += strs ;
        ptr_a = ptr_a.add(lda * strs);
    }
    // calculate edge (tail)
    if row_counter < m  { 
        for  row in 0..m-row_counter {
            for col in 0..k  {
                *ptr_dst.add( col * strs + row) = *ptr_a.add(row * lda + col);
            }
        }
    }   
}

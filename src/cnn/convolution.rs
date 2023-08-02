use super::super::matrix::mul::matrix_mul;

fn im2col   (input:& [f32],input_deep:usize,input_h:usize,input_w:usize,
                kernel_h:usize,kernel_w:usize,stride_h:usize,stride_w:usize,pad_h:usize,pad_w:usize,
                dst:&mut Vec<f32>,dst_h:usize,dst_w:usize ) {
   
    let mut i = 0;
    for input_ch in 0..input_deep {
        for kernel_row in 0..kernel_h {
            for kernel_col in 0..kernel_w {
                for dst_row in 0..dst_h {
                    for dst_col in 0..dst_w {

                        let mut stride_y = dst_row * stride_h + kernel_row;
                        stride_y = stride_y.wrapping_sub(pad_h );
                        let mut strode_x = dst_col * stride_w  + kernel_col;
                        strode_x = strode_x.wrapping_sub( pad_w);
                            if stride_y >= 0 && stride_y < input_h && strode_x >=0 && strode_x < input_w {
                                dst[i] = input[(input_ch * input_w + stride_y)*input_w + strode_x]  
                            } else {
                                dst[i] = 0.0
                            }
                            i+=1           
                    }
                }
            }
        }
    }
    

}
///    # Requirements:
///    1) number of input image channels must be equal to the number of kernel channels.
///    2) number of output image channels must be equal to the number of kernel filters.
///    3) expected fn pointer `fn(f32) -> f32` for activation function 
///    # Example
///    ```
///    let (image_ch,image_h, image_w ) =  (1,4,4);
///    let image:Vec<f32>  = vec![ 10.0, 20.0, 5.0, 7.0,
///                                8.0,  7.0,  3.0, 2.0,
///                                15.0, 9.0,  8.0, 8.0,
///                                4.0,  2.0,  7.0, 5.0,];   
///    let mut res = vec![0.0f32;8];
///    let (kernel_h,kernel_w) = (2,2);
///    let (stride_h,stride_w) = (2,2);
///    let (pad_h,pad_w)       = (0,0);
///    let bias:Vec<f32> = vec![0.0f32;2];
///   
///    //kernel has format  = 2,1,2,2 (filter,channel, row,col )
///    let kernel:Vec<f32> = vec![ 0.0, 1.0,  // first filter
///                               1.0, 0.0,
///    
///                                1.0, 0.0, // second filter
///                                0.0, 1.0];
///
///    // buf_size = (dst_h * dst_w) * image_ch * kernel_h * kernel_w;                            
///    let buf_size =  (2 * 2) * image_ch * kernel_h * kernel_w;                           
///    let mut buf = vec![0.0f32;buf_size];
/// 
///    // output volume (2,2,2)
///    let expected:Vec<f32> = vec![28.0, 10.0, 
///                                 13.0, 15.0,
///
///                                 17.0, 7.0,
///                                 17.0, 13.0]; 
///   
///    aicore::cnn::convolution::convolution(&image, image_ch, image_h, image_w, 
///                                        &kernel, kernel_h, kernel_w,
///                                        stride_h, stride_w, pad_h, pad_w, 
///                                        &bias, &mut res, 2, &mut buf, None);
///   
///   assert_eq!(&res,&expected)
/// ```
pub fn convolution (input:&Vec<f32>,input_ch:usize,input_h:usize,input_w:usize,
                        kernel:&Vec<f32>,kernel_h:usize,kernel_w:usize, stride_h:usize,stride_w:usize, pad_h:usize,pad_w:usize,
                        bias:&Vec<f32>,dst:&mut Vec<f32>,dst_ch:usize,buf:&mut Vec<f32>,func:Option<fn(f32)->f32>) {
        
        let dst_h = (input_h +  pad_h - kernel_h) / stride_h + 1 ;
        let dst_w = (input_w + pad_w - kernel_w) / stride_w + 1; 
        let m = dst_ch;
        let n = dst_h * dst_w;
        let k = input_ch * kernel_h * kernel_w;
        //
        im2col(&input,input_ch,input_h,input_w,kernel_h,kernel_w,stride_h,stride_w,pad_h,pad_w,buf,dst_h,dst_w);
        matrix_mul(&kernel[0..],m,k,&buf[0..],k,n,&mut dst[0..]);
        
        if func.is_some() {
            let f = func.unwrap();
            for i in 0..dst_ch {
                for j in 0..n {
                    dst[i*n+j] = f(dst[i*n+j] + bias[i])                
                }
            }
            return;
        }
        
        for i in 0..dst_ch {
            for j in 0..n {
                dst[i*n+j] = dst[i*n+j] + bias[i]                
            }
        }
    }
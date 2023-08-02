use aicore;


#[test]
fn rotate_matrix_180() {
    let  src = vec![1.0f32, 2.0f32,  3.0f32,
                              4.0f32, 5.0f32,  6.0f32,
                              7.0f32, 8.0f32,  9.0f32,];

    let expected = vec![9.0f32, 8.0f32, 7.0f32, 
                                  6.0f32, 5.0f32, 4.0f32, 
                                  3.0f32, 2.0f32, 1.0f32];

    let mut dst  = vec![0.0f32;3 * 3]; 
    aicore::matrix::shape::rotate_matrix180(&src, 3, 3, &mut dst);
    assert_eq!(expected,dst)
}
#[test]
fn transpose_matrix() {
    let  src = vec![1.0f32, 2.0f32, 3.0f32,
                              4.0f32, 5.0f32, 6.0f32,
                              7.0f32, 8.0f32, 9.0f32,];

    let expected = vec![1.0f32, 4.0f32, 7.0f32,
                                  2.0f32, 5.0f32, 8.0f32,
                                  3.0f32, 6.0f32, 9.0f32,];

    let mut dst  = vec![0.0f32;3 * 3]; 
    aicore::matrix::shape::transpose_matrix(&src, 3, 3, &mut dst);
    assert_eq!(expected,dst)
}
#[test]
fn test_convolution () {
  
    let (image_ch,image_h, image_w ) =  (1,4,4);
    let image:Vec<f32>  = vec![ 10.0, 20.0, 5.0, 7.0,
                                8.0,  7.0,  3.0, 2.0,
                                15.0, 9.0,  8.0, 8.0,
                                4.0,  2.0,  7.0, 5.0,];
   
    
    let mut res = vec![0.0f32;8];

    let (kernel_h,kernel_w) = (2,2);
    let (stride_h,stride_w) = (2,2);
    let (pad_h,pad_w)       = (0,0);

    let bias:Vec<f32> = vec![0.0f32;2];
    // kernel has format  = 2,1,2,2 (filter,channel, row,col )
    let kernel:Vec<f32> = vec![ 0.0, 1.0,  // first filter
                                1.0, 0.0,
    
                                1.0, 0.0, // second filter
                                0.0, 1.0];

    // buf_size = (dst_h * dst_w) * image_ch * kernel_h * kernel_w;                            
    let buf_size =  (2 * 2) * image_ch * kernel_h * kernel_w;                           
    let mut buf = vec![0.0f32;buf_size];
    // output volume (2,2,2)
    let expected:Vec<f32> = vec![28.0, 10.0, 
                                 13.0, 15.0,

                                 17.0, 7.0,
                                 17.0, 13.0]; 
   
    aicore::cnn::convolution::convolution(&image, image_ch, image_h, image_w, 
                                        &kernel, kernel_h, kernel_w,
                                        stride_h, stride_w, pad_h, pad_w, 
                                        &bias, &mut res, 2, &mut buf, None);

   assert_eq!(&res,&expected)
}


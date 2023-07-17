use ncore;


#[test]
fn rotate_matrix_180() {
    let  src = vec![1.0f32, 2.0f32,  3.0f32,
                              4.0f32, 5.0f32,  6.0f32,
                              7.0f32, 8.0f32,  9.0f32,];

    let expected = vec![9.0f32, 8.0f32, 7.0f32, 
                                  6.0f32, 5.0f32, 4.0f32, 
                                  3.0f32, 2.0f32, 1.0f32];

    let mut dst  = vec![0.0f32;3 * 3]; 
    ncore::matrix::shape::rotate_matrix180(&src, 3, 3, &mut dst);
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
    ncore::matrix::shape::transpose_matrix(&src, 3, 3, &mut dst);
    assert_eq!(expected,dst)
}